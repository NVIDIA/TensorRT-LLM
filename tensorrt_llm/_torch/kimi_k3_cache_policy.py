# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment-controlled KV-cache policies for Kimi K3."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from tensorrt_llm.models.modeling_utils import QuantConfig

KIMI_K3_FP8_KV_LAYERS_ENV = "TLLM_KIMI_K3_FP8_KV_LAYERS"
# Replaced by KIMI_K3_FP8_KV_LAYERS_ENV; rejected so that an old launch script
# cannot silently fall back to the NVFP4 default selection.
_REMOVED_KIMI_K3_BF16_KV_LAYERS_ENV = "TLLM_KIMI_K3_BF16_KV_LAYERS"


def _get_kimi_k3_text_config(pretrained_config: object) -> object:
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is not None:
        return text_config
    return pretrained_config


def get_kimi_k3_fp8_kv_layer_ids(
    pretrained_config: object, *, nvfp4_enabled: bool = False
) -> frozenset[int]:
    """Return zero-based Kimi K3 MLA layers that use FP8 KV cache.

    NVFP4 defaults to the last two MLA layers in the global layer schedule.
    An explicit environment value replaces this default; an empty value
    disables FP8 fallback. Explicit selections retain their validation at
    the model call site when NVFP4 is disabled.
    """
    if os.environ.get(_REMOVED_KIMI_K3_BF16_KV_LAYERS_ENV) is not None:
        raise ValueError(
            f"{_REMOVED_KIMI_K3_BF16_KV_LAYERS_ENV} is no longer supported: the "
            "Kimi K3 NVFP4 fallback layers now use FP8 KV cache. Set "
            f"{KIMI_K3_FP8_KV_LAYERS_ENV} instead."
        )
    raw_layer_ids = os.environ.get(KIMI_K3_FP8_KV_LAYERS_ENV)
    if raw_layer_ids is not None:
        raw_layer_ids = raw_layer_ids.strip()
        if not raw_layer_ids:
            return frozenset()
    elif not nvfp4_enabled:
        return frozenset()

    config = _get_kimi_k3_text_config(pretrained_config)
    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None) or getattr(
        pretrained_config, "architectures", None
    )
    is_kimi_k3 = model_type in ("kimi_k3", "kimi_k3_text", "kimi_linear") or any(
        "KimiK3" in architecture or "KimiLinear" in architecture
        for architecture in (architectures or ())
    )
    if not is_kimi_k3:
        return frozenset()

    linear_attn_config = getattr(config, "linear_attn_config", None)
    if linear_attn_config is None:
        raise ValueError(f"{KIMI_K3_FP8_KV_LAYERS_ENV} requires Kimi K3's linear_attn_config.")
    full_attn_layers = linear_attn_config.get("full_attn_layers", ())
    mla_layer_ids = {int(layer_number) - 1 for layer_number in full_attn_layers}
    if raw_layer_ids is None:
        layer_ids = frozenset(sorted(mla_layer_ids)[-2:])
    else:
        values = raw_layer_ids.split(",")
        if any(not value.strip() for value in values):
            raise ValueError(
                f"{KIMI_K3_FP8_KV_LAYERS_ENV} must be a comma-separated list "
                f"of zero-based layer IDs, got {raw_layer_ids!r}."
            )
        try:
            layer_ids = frozenset(int(value.strip()) for value in values)
        except ValueError as error:
            raise ValueError(
                f"{KIMI_K3_FP8_KV_LAYERS_ENV} must contain only integer "
                f"zero-based layer IDs, got {raw_layer_ids!r}."
            ) from error

    num_hidden_layers = int(config.num_hidden_layers)
    out_of_range = sorted(
        layer_id for layer_id in layer_ids if not 0 <= layer_id < num_hidden_layers
    )
    if out_of_range:
        raise ValueError(
            f"{KIMI_K3_FP8_KV_LAYERS_ENV} contains layer IDs outside "
            f"[0, {num_hidden_layers}): {out_of_range}."
        )

    non_mla_layers = sorted(layer_ids - mla_layer_ids)
    if non_mla_layers:
        raise ValueError(
            f"{KIMI_K3_FP8_KV_LAYERS_ENV} can select only MLA layers; "
            f"these layers use KDA instead: {non_mla_layers}."
        )

    return layer_ids


def resolve_kimi_k3_mla_kv_cache_quant_config(
    pretrained_config: object,
    layer_idx: Optional[int],
    quant_config: Optional[QuantConfig],
) -> Optional[QuantConfig]:
    """Return the attention quant config of one MLA layer after the FP8 fallback.

    The FP4 MLA cache manager stores the layers selected by
    :func:`get_kimi_k3_fp8_kv_layer_ids` in an FP8 pool. Their attention
    backends must therefore run FP8 KV cache attention. Every other layer, and
    every config whose KV cache is not NVFP4, is returned unchanged.

    The call is idempotent: a layer whose config already selects FP8 KV cache
    is returned unchanged. An explicit FP8 selection otherwise requires NVFP4
    KV cache for the model and is rejected for any other KV cache dtype.
    """
    if layer_idx is None or pretrained_config is None:
        return quant_config
    kv_quant_algo = quant_config.kv_cache_quant_algo if quant_config is not None else None
    nvfp4_enabled = kv_quant_algo == QuantAlgo.NVFP4
    fp8_layer_ids = get_kimi_k3_fp8_kv_layer_ids(pretrained_config, nvfp4_enabled=nvfp4_enabled)
    if layer_idx not in fp8_layer_ids or kv_quant_algo == QuantAlgo.FP8:
        return quant_config
    if not nvfp4_enabled:
        raise ValueError(
            f"{KIMI_K3_FP8_KV_LAYERS_ENV} requires the model-wide KV cache dtype "
            f"to be NVFP4, got {kv_quant_algo}."
        )
    # Build a new config: QuantConfig caches its derived quant modes, so a copy
    # with an updated field would keep reporting NVFP4.
    fields = {name: getattr(quant_config, name) for name in type(quant_config).model_fields}
    fields["kv_cache_quant_algo"] = QuantAlgo.FP8
    return type(quant_config)(**fields)


__all__ = [
    "KIMI_K3_FP8_KV_LAYERS_ENV",
    "get_kimi_k3_fp8_kv_layer_ids",
    "resolve_kimi_k3_mla_kv_cache_quant_config",
]
