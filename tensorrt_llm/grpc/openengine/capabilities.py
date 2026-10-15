# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this engine can actually do, defined once for Generate and Control.

Control advertises from this table and Generate enforces it, so a client can
never discover a capability that Generate then rejects.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Collection, Mapping

from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    BaseMultimodalInputProcessor,
)

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
    """Immutable media capability snapshot shared by Control and Generate."""

    placeholder_templates: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "placeholder_templates", MappingProxyType(dict(self.placeholder_templates))
        )

    @property
    def modalities(self) -> frozenset[str]:
        """Modalities the loaded model and request mapper both support."""
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


def resolve_media_capabilities(
    llm: Any, implemented_modalities: Collection[str]
) -> MediaCapabilities:
    """Resolve media support implemented by both the mapper and loaded model.

    A registry entry alone describes what this TensorRT-LLM build knows about;
    it does not prove that the loaded engine instantiated a multimodal input
    processor. Requiring both prevents a text-only model from advertising
    media merely because another image model is registered in the process. The
    request mapper supplies ``implemented_modalities`` from its handler table,
    so discovery cannot advertise a modality without a request implementation.
    """
    input_processor = getattr(llm, "input_processor", None)
    if not isinstance(input_processor, BaseMultimodalInputProcessor):
        return MediaCapabilities({})
    model_type = _loaded_model_type(llm)
    if not model_type:
        return MediaCapabilities({})

    placeholders = {
        modality: MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(model_type, modality)
        for modality in implemented_modalities
        if MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(model_type, modality)
    }
    # Validate templates once at startup. A malformed registry entry is a
    # server configuration failure, not a client request error discovered only
    # after Control has advertised the modality.
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
]
