# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment-controlled KV-cache policies for Kimi K3."""

from __future__ import annotations

import os

KIMI_K3_FP8_KV_LAYERS_ENV = "TLLM_KIMI_K3_FP8_KV_LAYERS"


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


__all__ = [
    "KIMI_K3_FP8_KV_LAYERS_ENV",
    "get_kimi_k3_fp8_kv_layer_ids",
]
