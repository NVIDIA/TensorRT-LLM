# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared capability resolution for OpenEngine Control and Generate."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Collection, Mapping

from tensorrt_llm.inputs.model_config import resolve_top_level_model_type
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    BaseMultimodalInputProcessor,
)

from .kv_events import events_config

# Guides each grammar backend can build, keyed by the `GuidedDecoding` oneof
# field name. Control advertises from this table and Generate enforces it, so
# the two cannot disagree. llguidance has no structural-tag matcher.
GUIDE_SUPPORT_BY_BACKEND: dict[str, frozenset[str]] = {
    "xgrammar": frozenset(
        {"json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"}
    ),
    "llguidance": frozenset({"json_schema", "regex", "ebnf_grammar", "json_object"}),
}


@dataclass(frozen=True)
class MediaCapabilities:
    """Media support shared by Control and Generate."""

    placeholder_templates: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "placeholder_templates", MappingProxyType(dict(self.placeholder_templates))
        )

    def __hash__(self) -> int:
        return hash(frozenset(self.placeholder_templates.items()))

    @property
    def modalities(self) -> frozenset[str]:
        """Modalities supported by both the model and request mapper."""
        return frozenset(self.placeholder_templates)

    def placeholder(self, modality: str, item_number: int) -> str:
        """Format the model placeholder for a one-based media item."""
        return self.placeholder_templates[modality].format(item_number)


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
    config = getattr(input_processor, "config", None) or getattr(llm, "_hf_model_config", None)
    if config is None:
        return ""
    return resolve_top_level_model_type(config)


def supports_image_routing(llm: Any, modalities: Collection[str]) -> bool:
    """Whether the loaded model exposes the validated image KV-event contract."""
    processor_config = getattr(getattr(llm, "input_processor", None), "config", None)
    if "image" not in modalities or resolve_top_level_model_type(processor_config) != "qwen3_vl":
        return False
    cache = getattr(llm.args, "kv_cache_config", None)
    return (
        getattr(llm.args, "cache_transceiver_config", None) is None
        and getattr(cache, "use_kv_cache_manager_v2", False) is True
        and getattr(cache, "enable_block_reuse", False) is True
        and events_config(llm) is not None
    )


def resolve_media_capabilities(
    llm: Any, implemented_modalities: Collection[str]
) -> MediaCapabilities:
    """Intersect model support with implemented request handlers.

    A live registered processor prevents registry-only false positives.
    """
    input_processor = getattr(llm, "input_processor", None)
    if getattr(getattr(llm, "args", None), "disable_mm_encoder", False):
        return MediaCapabilities({})

    model_type = _loaded_model_type(llm)
    if not model_type:
        return MediaCapabilities({})
    if not isinstance(input_processor, BaseMultimodalInputProcessor):
        registered_model_type = getattr(type(input_processor), "_registered_model_type", None)
        if registered_model_type != model_type:
            return MediaCapabilities({})
        if not callable(input_processor):
            return MediaCapabilities({})

    supported_modalities = getattr(input_processor, "supported_modalities", frozenset())
    placeholders = {
        modality: MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(model_type, modality)
        for modality in implemented_modalities
        if modality in supported_modalities
        and MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(model_type, modality)
    }
    # Fail startup before Control advertises a malformed template.
    for modality, template in placeholders.items():
        try:
            template.format(1)
        except (IndexError, KeyError, ValueError) as error:
            raise RuntimeError(
                f"invalid multimodal placeholder for model type '{model_type}' "
                f"and modality '{modality}': {error}"
            ) from error
    return MediaCapabilities(placeholders)


__all__ = [
    "GUIDE_SUPPORT_BY_BACKEND",
    "MediaCapabilities",
    "resolve_media_capabilities",
    "supported_guides",
    "supports_image_routing",
]
