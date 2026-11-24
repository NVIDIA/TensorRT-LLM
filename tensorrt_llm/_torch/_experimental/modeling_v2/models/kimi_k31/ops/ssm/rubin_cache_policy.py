# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 MLA layers whose latent KV is stored in FP8 beside an NVFP4 KV cache.

The selection is the ``fp8_kv_layers`` argument a target binds: the target builds its MLA layers
with it and the family cache manager receives it at construction, so both select the same layers.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: The construction argument that selects the layers.
FIELD = "fp8_kv_layers"


def _get_kimi_k31_text_config(pretrained_config: object) -> object:
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is not None:
        return text_config
    return pretrained_config


def get_kimi_k31_fp8_kv_layer_ids(
    pretrained_config: object,
    *,
    nvfp4_enabled: bool = False,
    fp8_kv_layers: Optional[Tuple[int, ...]] = None,
) -> frozenset[int]:
    """Return zero-based Kimi K3.1 MLA layers that use FP8 KV cache.

    With ``fp8_kv_layers`` None, an NVFP4 KV cache keeps the last two MLA layers of the global
    layer schedule in FP8 and any other KV cache none. ``()`` selects none; explicit layer ids
    select those layers, and the call sites reject them without an NVFP4 KV cache.

    Raises:
        TypeError: ``fp8_kv_layers`` is neither None nor a tuple of ints.
        ValueError: An explicit layer id is out of range or not an MLA layer.
    """
    config = _get_kimi_k31_text_config(pretrained_config)
    if getattr(config, "dsa_config", None) is None:
        return frozenset()

    selected = fp8_kv_layers
    if selected is not None:
        if not isinstance(selected, tuple) or any(type(layer) is not int for layer in selected):
            raise TypeError(
                f"Kimi K3.1 cache: {FIELD} must be None or a tuple of zero-based layer IDs, "
                f"got {selected!r}."
            )
        if not selected:
            return frozenset()
    elif not nvfp4_enabled:
        return frozenset()

    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None) or getattr(
        pretrained_config, "architectures", None
    )
    is_kimi_k3 = model_type in ("kimi_k3", "kimi_k31", "kimi_k3_text", "kimi_linear") or any(
        "KimiK3" in architecture or "KimiLinear" in architecture
        for architecture in (architectures or ())
    )
    if not is_kimi_k3:
        return frozenset()

    linear_attn_config = getattr(config, "linear_attn_config", None)
    if linear_attn_config is None:
        raise ValueError(f"Kimi K3.1 cache: {FIELD} requires Kimi K3's linear_attn_config.")
    full_attn_layers = linear_attn_config.get("full_attn_layers", ())
    mla_layer_ids = {int(layer_number) - 1 for layer_number in full_attn_layers}
    if selected is None:
        layer_ids = frozenset(sorted(mla_layer_ids)[-2:])
    else:
        layer_ids = frozenset(selected)

    num_hidden_layers = int(config.num_hidden_layers)
    out_of_range = sorted(
        layer_id for layer_id in layer_ids if not 0 <= layer_id < num_hidden_layers
    )
    if out_of_range:
        raise ValueError(
            f"Kimi K3.1 cache: {FIELD} contains layer IDs outside "
            f"[0, {num_hidden_layers}): {out_of_range}."
        )

    non_mla_layers = sorted(layer_ids - mla_layer_ids)
    if non_mla_layers:
        raise ValueError(
            f"Kimi K3.1 cache: {FIELD} can select only MLA layers; "
            f"these layers use KDA instead: {non_mla_layers}."
        )

    return layer_ids


__all__ = [
    "FIELD",
    "get_kimi_k31_fp8_kv_layer_ids",
]
