# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose a Modeling V2 language model and its multimodal encoders."""

from __future__ import annotations

import inspect
from functools import wraps
from itertools import combinations
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.model_hooks import call_load_weights
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalEncoderSpec,
    MultimodalModelMixin,
    _pending_multimodal_preparation,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import PreparedLlmInputs

__all__ = ["MultimodalEncoderSpec", "MultimodalModelForCausalLM"]

# The loader and router read these flags from the composite class before construction.
_LANGUAGE_MODEL_CLASS_ATTRIBUTES = {
    "EPLB_MODES": frozenset(),
    "moe_impls_honored": frozenset(),
    "draft_moe_impls_honored": frozenset(),
    "supports_nvfp4_kv_cache": False,
}


def _check_custom_forward(forward):
    signature = inspect.signature(forward)
    defaults = signature.bind_partial()
    defaults.apply_defaults()
    default_num_contexts = defaults.arguments.get("num_context_requests")

    @wraps(forward)
    def checked(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs).arguments if args else {}
        params = kwargs.get("multimodal_params", bound.get("multimodal_params"))
        metadata = kwargs.get(
            "attn_metadata", bound.get("attn_metadata", args[0] if args else None)
        )
        if not params or _pending_multimodal_preparation.get() is self:
            return forward(self, *args, **kwargs)
        num_contexts = getattr(metadata, "num_contexts", None)
        if num_contexts is None:
            num_contexts = kwargs.get(
                "num_context_requests", bound.get("num_context_requests", default_num_contexts)
            )
        if not self.select_multimodal_params(
            params, len(params) if num_contexts is None else num_contexts
        ):
            return forward(self, *args, **kwargs)
        token = _pending_multimodal_preparation.set(self)
        try:
            result = forward(self, *args, **kwargs)
            if _pending_multimodal_preparation.get() is self:
                raise RuntimeError(
                    "A custom multimodal forward must call prepare_multimodal_inputs or prepare_multimodal_embeds"
                )
            return result
        finally:
            _pending_multimodal_preparation.reset(token)

    return checked


class _EncoderSpecs(dict[str, MultimodalEncoderSpec]):
    def __init__(self, specs):
        super().__init__(specs)
        prefixes = []
        modalities = set()
        for name, spec in self.items():
            if (
                not spec.modalities
                or modalities.intersection(spec.modalities)
                or len(set(spec.modalities)) != len(spec.modalities)
            ):
                raise ValueError("Each modality must belong to one multimodal encoder")
            modalities.update(spec.modalities)
            weights = (
                (spec.weight_prefix,) if isinstance(spec.weight_prefix, str) else spec.weight_prefix
            )
            if not weights or not all(weights):
                raise ValueError(f"Encoder {name!r} requires a nonempty weight prefix")
            prefixes.extend(weights)
            tower = spec.cls
            if (
                not issubclass(tower, nn.Module)
                or inspect.isabstract(tower)
                or tower.forward is nn.Module.forward
                or any(
                    not callable(hook) or getattr(hook, "__isabstractmethod__", False)
                    for hook in (tower.forward, getattr(tower, "build_batched_input", None))
                )
            ):
                raise TypeError(f"Encoder {name!r} must implement build_batched_input and forward")
            if hasattr(tower, "get_mm_max_tokens_per_item") != hasattr(tower, "get_dummy_mm_data"):
                raise TypeError(
                    f"Encoder {name!r} must implement item sizing and dummy inputs together"
                )
        if any(
            left.startswith(right) or right.startswith(left)
            for left, right in combinations(prefixes, 2)
        ):
            raise ValueError("Multimodal encoder weight prefixes must not overlap")

    # nn.Module stores registered children in _modules. Let the instance see its
    # registered ModuleDict under the same name as the class's declarations.
    def __get__(self, instance, owner):
        return self if instance is None else instance._modules["mm_encoders"]


class MultimodalModelForCausalLM(MultimodalModelMixin, nn.Module):
    """Compose ``llm_class`` with the towers declared in ``mm_encoders``.

    Each tower packs compatible requests with ``build_batched_input``. Its forward
    returns [num_mm_tokens, hidden] in the same modality-major order. By default
    the mixin fuses these embeddings into the language model input. Set
    ``llm_requires_input_ids`` to pass the original token IDs, ``mm_embeds`` and
    the engine's ``mm_token_indices`` to the language model.
    """

    _inputs_embeds_conflict = (
        (AssertionError, "the engine passed inputs_embeds for a media step") if __debug__ else None
    )
    llm_class: ClassVar[type[DecoderModelForCausalLM]]
    mm_encoders: ClassVar[dict[str, MultimodalEncoderSpec]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(cls.mm_encoders, _EncoderSpecs):
            cls.mm_encoders = _EncoderSpecs(cls.mm_encoders)
        specs = cls.mm_encoders

        if not cls.llm_requires_input_ids and (
            not callable(cls._fuse_multimodal_embeddings)
            or getattr(cls._fuse_multimodal_embeddings, "__isabstractmethod__", False)
        ):
            raise TypeError("A V2 model that fuses embeddings needs a concrete fusion hook")
        if cls.encode_multimodal_inputs is not MultimodalModelMixin.encode_multimodal_inputs:
            raise TypeError("V2 encoders must use the grouped encode_multimodal_inputs template")
        if (
            cls.prepare_multimodal_inputs is not MultimodalModelMixin.prepare_multimodal_inputs
            or cls.prepare_multimodal_embeds is not MultimodalModelMixin.prepare_multimodal_embeds
        ):
            raise TypeError("V2 models must use shared multimodal input preparation")
        if cls.forward is not MultimodalModelMixin.forward and not any(
            issubclass(base, MultimodalModelForCausalLM) and base.forward is cls.forward
            for base in cls.__bases__
        ):
            cls.forward = _check_custom_forward(cls.forward)

        for flag in (
            "supports_encoder_cache",
            "supports_mm_encoder_item_scheduling",
            "supports_vision_gpu_normalize",
        ):
            if "mm_encoders" not in cls.__dict__ and flag not in cls.__dict__:
                continue
            required = [
                spec.cls
                for spec in specs.values()
                if flag != "supports_vision_gpu_normalize"
                or {"image", "video"}.intersection(spec.modalities)
            ]
            supported = bool(required) and all(getattr(tower, flag, False) for tower in required)
            if cls.__dict__.get(flag) and not supported:
                raise ValueError(f"{flag} requires support from all participating encoders")
            if flag not in cls.__dict__:
                setattr(cls, flag, supported)

        cls.supported_encoder_attention_backends = {
            name: getattr(spec.cls, "supported_encoder_attention_backends", frozenset())
            for name, spec in specs.items()
        }
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
        self.config = self.llm.config
        self.mm_encoders = nn.ModuleDict()
        groups = []
        if not model_config.disable_mm_encoder:
            for name, spec in type(self).mm_encoders.items():
                kwargs = {}
                if self.supported_encoder_attention_backends[name]:
                    kwargs["model_config"] = (
                        model_config.copy_with().with_encoder_attention_backend(name)
                    )
                tower_config = getattr(composite, f"{name}_config", composite)
                tower = spec.cls(tower_config, self.llm.model_config.torch_dtype, **kwargs)
                self.mm_encoders[name] = tower
                groups.append(EncoderGroup(spec.modalities, tower, tower.build_batched_input))
        self.mm_encoder_groups = tuple(groups)
        # Publish the text config for KV sizing only after every tower has accepted the deployment.
        model_config.pretrained_config = self.config

    @property
    def language_model(self) -> DecoderModelForCausalLM:
        return self.llm

    @property
    def text_embedding_layer(self) -> nn.Module:
        return self.llm.model.embed_tokens

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

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
        """Route checkpoint keys by tower prefix, skipping values for disabled towers in this mapping."""
        encoder_keys = set()
        for name, spec in type(self).mm_encoders.items():
            keys = [key for key in weights.keys() if key.startswith(spec.weight_prefix)]
            if name in self.mm_encoders:
                self.mm_encoders[name].load_weights({key: weights[key] for key in keys})
            if isinstance(weights, ConsumableWeightsDict):
                weights.mark_consumed_keys(keys)
            else:
                encoder_keys.update(keys)
        # Shard-aware decoders need the loader's checkpoint metadata and handles.
        if not isinstance(weights, ConsumableWeightsDict):
            weights = {key: weights[key] for key in weights.keys() if key not in encoder_keys}
        if (
            weight_mapper is not None
            and "weight_mapper" in inspect.getfullargspec(self.llm.load_weights).args
        ):
            weight_mapper.init_model_and_config(self.llm, self.llm.model_config)
        call_load_weights(self.llm.load_weights, weights, weight_mapper)
