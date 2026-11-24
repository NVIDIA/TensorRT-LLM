# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine facts for Kimi K3.1 that are read before a target is routed.

Listed in ``_architecture_index`` for model_type ``kimi_k31``. Nothing here builds a model: it tells the engine how to
read the config, which input processor turns a multimodal prompt into token ids, and the OpenAI server how to serve
the model. Anything heavier than the config classes is imported inside the function that needs it.
"""

from __future__ import annotations

from .config import KimiK31Config

CONFIG_CLASSES = {KimiK31Config.model_type: KimiK31Config}

#: K3.1 writes tool calls and thinking in K3's markup (the same XTML channels), so ``trtllm-serve`` reads them with the
#: K3 parsers when ``--tool_parser`` / ``--reasoning_parser`` is ``auto`` or not given.
TOOL_PARSER = "kimi_k3"
REASONING_PARSER = "kimi_k3"


def load_config(config_dict: dict, **kwargs) -> KimiK31Config:
    """The composite config the engine serves for a ``kimi_k31`` config.json.

    The composite is kept, text-only (``language_model_only``) or not: it carries the vision tower's fields and the
    text config's DSA indexer, and ``architectures`` names the vision-language wrapper, which is what routing keys on.
    The flattened Kimi text model has no DSA indexer, so a K3.1 config never becomes one.
    """
    config = KimiK31Config.from_dict(config_dict, **kwargs)
    config.architectures = ["KimiK31ForConditionalGeneration"]
    return config


def input_processor(config):
    """The K3 image input processor (K3.1's vision input is K3's), registered with the prompt placeholders of
    model_type ``kimi_k31``. The driver process builds it without building the model."""
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    from .input_processor import KIMI_K31_PLACEHOLDER_METADATA, K31InputProcessor

    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        KimiK31Config.model_type, KIMI_K31_PLACEHOLDER_METADATA, registrant_module=__name__
    )
    K31InputProcessor._registered_model_type = KimiK31Config.model_type
    return K31InputProcessor


def serving_extension():
    """K3's serving extension: K3.1 has K3's API semantics (message-level tools, the pinned sampling parameters, the
    tool-call grammar) and its chat template renders the same thinking efforts (low / high / max)."""
    from tensorrt_llm.serve.extensions.kimi_k3 import KimiK3ServingExtension

    return KimiK3ServingExtension()
