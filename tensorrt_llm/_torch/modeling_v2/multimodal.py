# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose a Modeling V2 language model and its multimodal encoders."""

from __future__ import annotations

import inspect
from itertools import combinations
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.checkpoints.model_hooks import call_load_weights
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderSpec,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import PreparedLlmInputs

__all__ = ["MultimodalEncoderSpec", "MultimodalModelForCausalLM"]

# The loader and router read these flags from the composite class before construction.
_LANGUAGE_MODEL_CLASS_ATTRIBUTES = {
    "moe_impls_honored": frozenset(),
    "draft_moe_impls_honored": frozenset(),
    "supports_nvfp4_kv_cache": False,
}


class _EncoderSpecs(dict[str, MultimodalEncoderSpec]):
    # nn.Module stores registered children in _modules. Let the instance see its
    # registered ModuleDict under the same name as the class's declarations.
    def __get__(self, instance, owner):
        return self if instance is None else instance._modules["mm_encoders"]


class MultimodalModelForCausalLM(MultimodalModelMixin, nn.Module):
    """Compose ``llm_class`` with the towers declared in ``mm_encoders``.

    A tower takes the vision config and activation dtype. Its forward accepts one
    MultimodalParams and returns [num_mm_tokens, hidden] for its modalities in
    prompt order. By default the mixin fuses these embeddings into the language
    model input. Set ``llm_requires_input_ids`` to pass the original token IDs,
    ``mm_embeds`` and the engine's ``mm_token_indices`` to the language model.
    """

    _inputs_embeds_conflict = (
        (AssertionError, "the engine passed inputs_embeds for a media step") if __debug__ else None
    )
    llm_class: ClassVar[type[DecoderModelForCausalLM]]
    mm_encoders: ClassVar[dict[str, MultimodalEncoderSpec]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(cls.mm_encoders, _EncoderSpecs):
            for left, right in combinations(cls.mm_encoders.values(), 2):
                if set(left.modalities).intersection(right.modalities):
                    raise ValueError("Each modality must belong to one multimodal encoder")
                if left.weight_prefix.startswith(
                    right.weight_prefix
                ) or right.weight_prefix.startswith(left.weight_prefix):
                    raise ValueError("Multimodal encoder weight prefixes must not overlap")
            cls.mm_encoders = _EncoderSpecs(cls.mm_encoders)
        llm_class = cls.__dict__.get("llm_class")
        for name, default in _LANGUAGE_MODEL_CLASS_ATTRIBUTES.items():
            if llm_class is not None and name not in cls.__dict__:
                setattr(cls, name, getattr(llm_class, name, default))

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
        return cls.llm_class.get_preferred_kv_cache_manager_version(pretrained_config)

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        composite = model_config.pretrained_config
        if not isinstance(composite, PretrainedConfig):
            raise ValueError("model_config.pretrained_config must be a PretrainedConfig")
        # The engine's own `ModelConfig`: the loader, the engine and KV sizing read it here.
        self.model_config = model_config
        # Built as the engine builds a decoder model (`skip_create_weights_in_init`), on its own
        # shallow copy, where it adopts the text config.
        self.llm = self.llm_class(model_config.copy_with(skip_create_weights_in_init=True))
        # Everything that sizes the pools reads the text config.
        model_config.pretrained_config = self.llm.config
        self.config = self.llm.config
        self.mm_encoders = nn.ModuleDict(
            {
                name: spec.cls(composite.vision_config, self.dtype)
                for name, spec in type(self).mm_encoders.items()
                if not model_config.disable_mm_encoder
            }
        )

    @property
    def language_model(self) -> DecoderModelForCausalLM:
        return self.llm

    @property
    def text_embedding_layer(self) -> nn.Module:
        return self.llm.model.embed_tokens

    @property
    def mm_encoder(self) -> nn.Module | None:
        """The first local tower for runtime checks that use the single-encoder attribute."""
        return next(iter(self.mm_encoders.values()), None)

    @property
    def dtype(self) -> torch.dtype:
        """The activation dtype. Set, not read off the first parameter, which may be a
        quantized one."""
        return self.model_config.torch_dtype

    # What the engine reads off the routed model is the language model's.

    @property
    def multimodal_token_ids(self):
        return getattr(self.llm, "multimodal_token_ids", None)

    @property
    def lm_head(self) -> nn.Module:
        return self.llm.lm_head

    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def spec_worker(self):
        return self.llm.spec_worker

    def apply_llm_torch_compile(self, *, backend, fullgraph: bool) -> None:
        """Compile only the language model's core; the tower stays eager."""
        self.llm.model = torch.compile(self.llm.model, backend=backend, fullgraph=fullgraph)

    def validate_request(self, request) -> None:
        """Check the decoder vocabulary before the towers validate the prompt."""
        num_embeddings = self.llm.lm_head.num_embeddings
        token_bound = request.py_prompt_token_bound
        if token_bound is not None:
            if token_bound >= num_embeddings:
                raise ValueError("Token ID out of range")
        elif not request.check_token_id_range(num_embeddings):
            raise ValueError("Token ID out of range")
        super().validate_request(request)

    def get_language_model_forward_kwargs(
        self, *, raw_input_ids: torch.Tensor | None, mm_inputs: PreparedLlmInputs, **kwargs: Any
    ) -> dict[str, Any]:
        """Pass engine arguments through to the V2 decoder and apply model-specific overrides."""
        prepared = super().get_language_model_forward_kwargs(
            raw_input_ids=raw_input_ids, mm_inputs=mm_inputs, **kwargs
        )
        # These two arguments describe the step to the multimodal hooks.
        kwargs.pop("multimodal_params", None)
        kwargs.pop("num_generation_requests", None)
        return {**kwargs, **prepared}

    def load_weights(self, weights, weight_mapper=None) -> None:
        """Route checkpoint keys by tower prefix. Disabled towers leave their weights unread."""
        encoder_weights = {name: {} for name in self.mm_encoders}
        text_weights = {}
        for key in weights.keys():
            for name, spec in type(self).mm_encoders.items():
                if key.startswith(spec.weight_prefix):
                    if name in encoder_weights:
                        encoder_weights[name][key] = weights[key]
                    break
            else:
                text_weights[key] = weights[key]
        for name, tower in self.mm_encoders.items():
            tower.load_weights(encoder_weights[name])
        if (
            weight_mapper is not None
            and "weight_mapper" in inspect.getfullargspec(self.llm.load_weights).args
        ):
            weight_mapper.init_model_and_config(self.llm, self.llm.model_config)
        call_load_weights(self.llm.load_weights, text_weights, weight_mapper)
