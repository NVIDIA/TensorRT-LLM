# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this engine can actually do, defined once for Generate and Control.

Control advertises from this table and Generate enforces it, so a client can
never discover a capability that Generate then rejects.
"""

from typing import Any

from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    BaseMultimodalInputProcessor,
)

# OpenEngine support is intentionally narrower than the LLM API today. Add a
# modality here only once request mapping and conformance coverage exist for it.
_OPENENGINE_MEDIA_MODALITIES = ("image",)

# Guides each grammar backend can build, keyed by the `GuidedDecoding` oneof
# field name. Control advertises from this table and Generate enforces it, so
# the two cannot disagree. llguidance has no structural-tag matcher.
GUIDE_SUPPORT_BY_BACKEND: dict[str, frozenset[str]] = {
    "xgrammar": frozenset(
        {"json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"}
    ),
    "llguidance": frozenset({"json_schema", "regex", "ebnf_grammar", "json_object"}),
}


def supported_guides(backend: str | None) -> frozenset[str]:
    """Guides `backend` can build; empty when none can be.

    With no backend configured the engine never builds a grammar and drops the
    per-request guided params silently, so accepting the request would return
    unconstrained text as a success. Failing closed here also keeps Generate
    aligned with GetModelInfo, which reports guided decoding unsupported for
    that engine.
    """
    if not backend:
        return frozenset()
    return GUIDE_SUPPORT_BY_BACKEND.get(backend.lower(), frozenset())


def _loaded_model_type(llm: Any) -> str:
    """Return the top-level model type for the loaded input processor."""
    input_processor = getattr(llm, "input_processor", None)
    config = (
        input_processor.config
        if isinstance(input_processor, BaseMultimodalInputProcessor)
        else getattr(llm, "_hf_model_config", None)
    )
    if config is None:
        return ""

    # Composite Hugging Face configs can delegate the instance attribute to
    # their text config. The class attribute retains the top-level registry key.
    model_type = getattr(type(config), "model_type", None) or getattr(config, "model_type", "")
    return model_type if isinstance(model_type, str) else ""


def supported_media_modalities(llm: Any) -> frozenset[str]:
    """OpenEngine media modalities supported by the loaded model.

    A registry entry alone describes what this TensorRT-LLM build knows about;
    it does not prove that the loaded engine instantiated a multimodal input
    processor. Requiring both prevents a text-only model from advertising
    media merely because another image model is registered in the process.
    """
    input_processor = getattr(llm, "input_processor", None)
    if not isinstance(input_processor, BaseMultimodalInputProcessor):
        return frozenset()
    model_type = _loaded_model_type(llm)
    if not model_type:
        return frozenset()
    return frozenset(
        modality
        for modality in _OPENENGINE_MEDIA_MODALITIES
        if MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(model_type, modality)
    )


def media_placeholder(llm: Any, modality: str, item_number: int) -> str:
    """Return the model-specific placeholder for a one-based media item."""
    model_type = _loaded_model_type(llm)
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(model_type, modality).format(item_number)


__all__ = [
    "GUIDE_SUPPORT_BY_BACKEND",
    "media_placeholder",
    "supported_guides",
    "supported_media_modalities",
]
