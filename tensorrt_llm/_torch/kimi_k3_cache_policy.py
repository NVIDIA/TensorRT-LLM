# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment-controlled KV-cache policies for Kimi K3."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

KIMI_K3_FP8_KV_LAYERS_ENV = "TLLM_KIMI_K3_FP8_KV_LAYERS"
# Deprecated: it listed MLA layers that kept a higher-precision KV cache.
# It is still honored as the FP8 layer list (with a warning) when
# KIMI_K3_FP8_KV_LAYERS_ENV is unset, so existing deployments keep working.
_DEPRECATED_KIMI_K3_BF16_KV_LAYERS_ENV = "TLLM_KIMI_K3_BF16_KV_LAYERS"
# MLA layers the NVFP4 default keeps in FP8, counted from the end of the
# global layer schedule. Kimi K3's accuracy is most sensitive to KV-cache
# quantization error in its last MLA layers: an NVFP4 KV cache there
# measurably lowers accuracy, while an FP8 KV cache for just those layers
# recovers it at a small memory cost.
_DEFAULT_NUM_FP8_KV_LAYERS = 2


def _get_kimi_k3_text_config(pretrained_config: object) -> object:
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is not None:
        return text_config
    return pretrained_config


def is_kimi_k3(pretrained_config: object) -> bool:
    config = _get_kimi_k3_text_config(pretrained_config)
    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None) or getattr(
        pretrained_config, "architectures", None
    )
    return model_type in ("kimi_k3", "kimi_k3_text", "kimi_linear") or any(
        "KimiK3" in architecture or "KimiLinear" in architecture
        for architecture in (architectures or ())
    )


def _get_raw_fp8_kv_layer_ids(pretrained_config: object) -> Optional[str]:
    """Read the FP8 layer list, falling back to the deprecated variable."""
    raw_layer_ids = os.environ.get(KIMI_K3_FP8_KV_LAYERS_ENV)
    deprecated_layer_ids = os.environ.get(_DEPRECATED_KIMI_K3_BF16_KV_LAYERS_ENV)
    if deprecated_layer_ids is None or not is_kimi_k3(pretrained_config):
        return raw_layer_ids
    if raw_layer_ids is not None:
        logger.warning_once(
            f"{_DEPRECATED_KIMI_K3_BF16_KV_LAYERS_ENV} is deprecated and ignored "
            f"because {KIMI_K3_FP8_KV_LAYERS_ENV} is set; unset it.",
            key="kimi_k3_bf16_kv_layers_env_ignored",
        )
        return raw_layer_ids
    logger.warning_once(
        f"{_DEPRECATED_KIMI_K3_BF16_KV_LAYERS_ENV} is deprecated; its layers now "
        f"use an FP8 KV cache. Use {KIMI_K3_FP8_KV_LAYERS_ENV} instead.",
        key="kimi_k3_bf16_kv_layers_env_deprecated",
    )
    return deprecated_layer_ids


def _mla_layers_per_pp_stage(
    mla_layer_ids: set[int], num_hidden_layers: int, mapping: Optional[Mapping]
) -> dict[int, list[int]]:
    """Group MLA layers by the pipeline stage that owns them (one stage without a mapping)."""
    stages: dict[int, list[int]] = {}
    for layer_id in sorted(mla_layer_ids):
        pp_rank = 0 if mapping is None else mapping.pp_rank_of_layer(layer_id, num_hidden_layers)
        stages.setdefault(pp_rank, []).append(layer_id)
    return stages


def _default_fp8_kv_layer_ids(
    mla_layer_ids: set[int], num_hidden_layers: int, mapping: Optional[Mapping]
) -> frozenset[int]:
    """The last MLA layers, minus those whose stage would keep no NVFP4 MLA layer.

    The FP4 MLA cache layout needs at least one NVFP4 MLA layer on every
    pipeline stage that owns MLA layers. A stage whose MLA layers all fall
    in the default selection keeps its last one in FP8 and its earlier ones
    in NVFP4; a stage with one MLA layer keeps it in NVFP4. The result
    depends only on the model and the pipeline layout, so every rank and
    every caller resolves the same set.
    """
    layer_ids = set(sorted(mla_layer_ids)[-_DEFAULT_NUM_FP8_KV_LAYERS:])
    for stage_layer_ids in _mla_layers_per_pp_stage(
        mla_layer_ids, num_hidden_layers, mapping
    ).values():
        if layer_ids.issuperset(stage_layer_ids):
            layer_ids.difference_update(stage_layer_ids[:-1] or stage_layer_ids)
    return frozenset(layer_ids)


def get_kimi_k3_fp8_kv_layer_ids(
    pretrained_config: object,
    *,
    nvfp4_enabled: bool = False,
    mapping: Optional[Mapping] = None,
) -> frozenset[int]:
    """Return zero-based Kimi K3 MLA layers that use FP8 KV cache.

    NVFP4 defaults to the last two MLA layers in the global layer schedule,
    except that every pipeline stage owning MLA layers keeps at least one in
    NVFP4 (see ``_default_fp8_kv_layer_ids``). An explicit environment value
    replaces this default and is rejected if it leaves such a stage without
    an NVFP4 MLA layer; an empty value disables FP8 fallback. Explicit
    selections retain their validation at the model call site when NVFP4 is
    disabled.

    ``mapping`` supplies the pipeline layout; without it the model is
    treated as one stage. Every caller of one model must pass the same
    mapping so that they agree on the selection.
    """
    raw_layer_ids = _get_raw_fp8_kv_layer_ids(pretrained_config)
    if raw_layer_ids is not None:
        raw_layer_ids = raw_layer_ids.strip()
        if not raw_layer_ids:
            return frozenset()
    elif not nvfp4_enabled:
        return frozenset()

    if not is_kimi_k3(pretrained_config):
        return frozenset()

    config = _get_kimi_k3_text_config(pretrained_config)

    linear_attn_config = getattr(config, "linear_attn_config", None)
    if linear_attn_config is None:
        raise ValueError(f"{KIMI_K3_FP8_KV_LAYERS_ENV} requires Kimi K3's linear_attn_config.")
    full_attn_layers = linear_attn_config.get("full_attn_layers", ())
    num_hidden_layers = int(config.num_hidden_layers)
    # A truncated config may keep full_attn_layers entries past its last layer.
    mla_layer_ids = {
        int(layer_number) - 1
        for layer_number in full_attn_layers
        if 0 < int(layer_number) <= num_hidden_layers
    }
    if raw_layer_ids is None:
        return _default_fp8_kv_layer_ids(mla_layer_ids, num_hidden_layers, mapping)

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

    if nvfp4_enabled:
        for pp_rank, stage_layer_ids in sorted(
            _mla_layers_per_pp_stage(mla_layer_ids, num_hidden_layers, mapping).items()
        ):
            if layer_ids.issuperset(stage_layer_ids):
                raise ValueError(
                    f"{KIMI_K3_FP8_KV_LAYERS_ENV} selects every MLA layer of "
                    f"pipeline stage {pp_rank} ({stage_layer_ids}); an NVFP4 KV "
                    "cache needs at least one NVFP4 MLA layer per stage."
                )

    return layer_ids


__all__ = [
    "KIMI_K3_FP8_KV_LAYERS_ENV",
    "get_kimi_k3_fp8_kv_layer_ids",
]
