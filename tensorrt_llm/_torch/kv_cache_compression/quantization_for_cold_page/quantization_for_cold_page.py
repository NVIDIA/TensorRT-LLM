# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common runtime pipeline for cold-page quantization."""

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

# These models use contiguous RoPE: key-only buffers are MLA latents with RoPE
# trailing; key+value buffers are partial-rotary GQA with RoPE leading. Adding a
# model requires verifying that placement, not just its model_type. Interleaved
# RoPE or MLA caches exposing a value buffer need a separate layout rule.
# Other models ignore skip_rope_quantization.
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
_DEEPSEEK_V4_ROPE_DIM = _DEEPSEEK_V4_ROW_STRIDE - _DEEPSEEK_V4_NOPE_DIM
_DEEPSEEK_V4_FOOTER_SCALE_ROW_BYTES = 584


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Common codec registration, callbacks, and KV-row geometry for cold-page quantizers."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    _FORMAT_NAME = "NVFP4"
    """Format name used in user-facing messages."""
    _skip_rope_quantization = False
    _RANGE_ALIGNMENT = 16
    """Quantized ranges must start and end on multiples of this many elements."""
    _RANGE_ALIGNMENT_UNIT = "NVFP4 scale groups"

    @property
    def mixed_precision_high_window(self) -> Optional[tuple[int, int]]:
        """``(first_n, last_n)`` from the config, else None."""
        return getattr(self.config, "mixed_precision_high_window", None)

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
        """Create one native codec with state isolated to this KVCM."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state = self.build_codec_state(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
        return native.create_python_cold_page_codec(self, codec_state)

    def configure(self, codec_state: Any, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve hot buffers and publish each lifecycle's cold-page size."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state.lifecycle_metadata = tuple(
            self.build_lifecycle_metadata(codec_state, lifecycle) for lifecycle in lifecycles
        )
        properties = []
        for metadata in codec_state.lifecycle_metadata:
            lifecycle = native.ColdPageLifecycleProperties()
            lifecycle.cold_page_bytes = metadata.cold_page_bytes
            lifecycle.page_index_location = self._page_index_location(native)
            properties.append(lifecycle)
        return properties

    def _page_index_location(self, native: Any) -> object:
        """Where KVCM places the Page-index pairs passed to encode and decode."""
        return native.ColdPageIndexLocation.HOST

    def build_codec_state(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Build the format-specific state owned by one native codec."""
        raise NotImplementedError

    def build_lifecycle_metadata(self, codec_state: object, lifecycle: object) -> object:
        """Resolve one KVCM lifecycle into format-specific launch metadata."""
        raise NotImplementedError

    def _calculate_quantized_range(
        self,
        row_elements: int,
        *,
        skip_rope: bool,
        rope: tuple[int, int] | None = None,
        key_only: bool = False,
        buffer_name: str,
    ) -> tuple[int, int]:
        """Return (start, count) of the numbers of each K or V vector that get quantized."""

        if not skip_rope:
            return 0, row_elements
        if rope is None:
            text_config = self.pretrained_config.get_text_config()
            rope_parameters = getattr(text_config, "rope_parameters", None) or {}
            if not isinstance(rope_parameters, dict):
                raise ValueError(f"{buffer_name}: rope_parameters must be a mapping")
            if any(isinstance(value, dict) for value in rope_parameters.values()):
                raise NotImplementedError(
                    f"{buffer_name}: layer-specific rope_parameters require an explicit "
                    "cold-page RoPE layout"
                )
            # rotary_emb_base and rope_theta are rotation frequencies, never widths.
            rope_dim = None
            for width_field in ("rotary_dim", "qk_rope_head_dim"):
                width = rope_parameters.get(width_field, getattr(text_config, width_field, None))
                if width is not None:
                    if type(width) is not int:
                        raise ValueError(f"{buffer_name}: {width_field} must be an integer width")
                    rope_dim = width
                    break
            if rope_dim is None:
                rotary_factor = 1.0
                for factor_field in ("rotary_pct", "partial_rotary_factor"):
                    factor = rope_parameters.get(
                        factor_field, getattr(text_config, factor_field, None)
                    )
                    if factor is not None:
                        if type(factor) not in (int, float) or not math.isfinite(factor):
                            raise ValueError(
                                f"{buffer_name}: {factor_field} must be a finite rotary fraction"
                            )
                        rotary_factor = factor
                        break
                rope_dim = int(row_elements * rotary_factor)
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
                f"outside the {row_elements}-element row; check rotary_dim, qk_rope_head_dim, "
                "rotary_pct and partial_rotary_factor in the model config"
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
        alignment = self._RANGE_ALIGNMENT
        if start % alignment or elements % alignment:
            raise ValueError(
                f"{buffer_name}: the quantized range [{start}, {start + elements}) must start and "
                f"end on {alignment}-element {self._RANGE_ALIGNMENT_UNIT}"
            )
        return start, elements

    @staticmethod
    def _deepseek_v4_model_layers(
        attention_layers: Sequence["AttentionLayerConfig"], pp_layers: Sequence[int]
    ) -> tuple[dict[int, set[str]], bool, dict[int, int]]:
        """Classify DeepSeek-V4 KVCM layers by role and map them to their model layer.

        Returns the roles of every DeepSeek-V4 layer, whether a CSA cache exists, and (only then)
        the model layer each KVCM layer belongs to: every model layer starts with an SWA Page.
        """

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
        return roles_by_layer, has_csa_cache, model_layers

    def _deepseek_v4_csa_tokens_per_page(
        self, cache_config: object, layer: "AttentionLayerConfig", *, runtime_type: int
    ) -> int:
        """Validate a DeepSeek-V4 CSA Page and return its number of compressed rows."""

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
                f"{self._FORMAT_NAME} cold-page compression does not support DeepSeek-V4 "
                "fp8_ds_mla footer-scale Pages"
            )
        if configured_bytes != raw_bytes:
            raise ValueError(
                f"DeepSeek-V4 {_DEEPSEEK_V4_COMPRESS} buffer has "
                f"{configured_bytes} bytes; expected {raw_bytes} for an ordinary runtime Page"
            )
        return tokens_per_page
