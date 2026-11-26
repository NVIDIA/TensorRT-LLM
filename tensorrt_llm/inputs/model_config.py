# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for resolving model configuration metadata."""

from typing import Any


def resolve_top_level_model_type(model_config: Any) -> str:
    """Return the top-level registry model type for a loaded config.

    Composite configs can delegate the instance attribute to ``text_config``.
    The class attribute retains the top-level AutoConfig/registry key.
    """
    get_media_config = getattr(type(model_config), "get_media_config", None)
    if get_media_config is not None:
        # A decoder config flattened by the model loader can retain its composite.
        model_config = get_media_config(model_config) or model_config
    model_type = getattr(type(model_config), "model_type", None) or getattr(
        model_config, "model_type", ""
    )
    return model_type if isinstance(model_type, str) else ""
