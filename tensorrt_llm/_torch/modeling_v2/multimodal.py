# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared base class of Modeling V2 vision-language models.

A vision-language target is composite, as a Modeling V1 vision-language model is: a
:class:`ModelingV2ForConditionalGeneration` holds a Modeling V2 language model target as
``llm`` and its encoder tower as ``mm_encoder``. The shared multimodal runtime
(``MultimodalModelMixin``) selects the step's media requests, runs or reuses the tower
(encoder cache) and keeps the current chunk's rows, so the executor sees one multimodal model
shape for V1 and V2.

A target subclasses this base as ``ModelingV2<Target>ForConditionalGeneration`` and declares
what differs:

* ``language_model_class``: the language model target. It is built on a copy of the
  ``ModelConfig`` and adopts the text config on that copy, as it does when it is routed alone.
* ``mm_encoder_class``: the tower, an ``nn.Module`` and ``MultimodalEncoderMixin`` that
  declares and loads its own weights. The base builds it as
  ``mm_encoder_class(vision_config, dtype)``: the composite config's ``vision_config`` and the
  activation dtype. A tower built another way overrides ``build_mm_encoder`` instead. No tower
  is built with ``disable_mm_encoder``.
* ``mm_encoder_weight_prefix``: the checkpoint prefix of the tower's weights. The tower loads
  those keys and the language model every other key.
* ``encode_multimodal_inputs``: the tower's rows for the selected requests, in request order.
* ``language_model_takes_rows``: how the rows reach the language model. False (the default):
  the mixin fuses them into ``inputs_embeds``. True: ``input_ids`` stay as they are and the
  language model takes ``multimodal_rows=(rows, mm_token_indices)`` and places the rows itself;
  the engine then allows the breakable prefill CUDA graph, as for the language model alone.

Optional: ``validate_media_request`` (admission checks of the tower) and the mixin's feature
flags such as ``supports_encoder_cache``. To pass extra inputs to the language model, as V1
Qwen3-VL does for DeepStack, a target overrides ``get_language_model_extra_forward_kwargs``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.models.checkpoints.model_hooks import call_load_weights
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalModelMixin,
    PreparedLlmInputs,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.inputs.multimodal import has_media_or_embeddings

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig

__all__ = ["ModelingV2ForConditionalGeneration"]

#: `multimodal_data` entries the tower encodes, or rows an earlier step already made.
_ENCODED_KEYS = ("image", "video", "audio", "multimodal_embedding")

#: Language model flags that are read off the routed class, which is the composite, before any
#: model is built: `moe_impls_honored` and `draft_moe_impls_honored` by the router's MoE tactic
#: check (`_router_index._refuse_unhonored_moe_tactics`), `supports_nvfp4_kv_cache` by the
#: model loader's SM107 KV cache check. Each maps to the value its reader takes when a class has
#: none. A subclass that sets `language_model_class` takes that class's values, or these, unless
#: it sets its own, so it never keeps a parent's; without them the router would refuse a set MoE
#: tactic and the loader would switch an NVFP4 KV cache to FP8. Both read the class, so a
#: property cannot do this.
_LANGUAGE_MODEL_CLASS_ATTRIBUTES = {
    "moe_impls_honored": frozenset(),
    "draft_moe_impls_honored": frozenset(),
    "supports_nvfp4_kv_cache": False,
}


class ModelingV2ForConditionalGeneration(MultimodalModelMixin, PreTrainedModel):
    """A Modeling V2 language model target as ``llm`` and its encoder tower as ``mm_encoder``,
    behind the shared multimodal runtime. See the module docstring for what a target declares.

    Same composite shape as a Modeling V1 vision-language model (for example
    ``Qwen3VLModelBase``), holding only the code every V2 target needs.
    """

    language_model_class: ClassVar[type[DecoderModelForCausalLM]]
    mm_encoder_class: ClassVar[type[nn.Module] | None] = None
    mm_encoder_weight_prefix: ClassVar[str]
    language_model_takes_rows: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        llm_class = cls.__dict__.get("language_model_class")
        for name, default in _LANGUAGE_MODEL_CLASS_ATTRIBUTES.items():
            if llm_class is not None and name not in cls.__dict__:
                setattr(cls, name, getattr(llm_class, name, default))

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
        return cls.language_model_class.get_preferred_kv_cache_manager_version(pretrained_config)

    def _check_and_adjust_experts_implementation(self, *args, **kwargs):
        """No-op: the model holds no MoE layer of its own for the transformers check."""
        return None

    def __init__(self, model_config: ModelConfig):
        composite = model_config.pretrained_config
        super().__init__(composite)
        # The engine's own `ModelConfig`: the loader, the engine and KV sizing read it here.
        self.model_config = model_config
        # Built as the engine builds a decoder model (`skip_create_weights_in_init`), on its own
        # shallow copy, where it adopts the text config.
        self.llm = self.language_model_class(
            model_config.copy_with(skip_create_weights_in_init=True)
        )
        # Everything that sizes the pools reads the text config.
        model_config.pretrained_config = self.llm.config
        self.config = self.llm.config
        # None with `disable_mm_encoder`: no encoder weight is declared or loaded, and a
        # media request is refused at admission.
        self.mm_encoder = (
            None if model_config.disable_mm_encoder else self.build_mm_encoder(composite)
        )

    def build_mm_encoder(self, pretrained_config: PretrainedConfig) -> nn.Module | None:
        """The encoder tower of the composite ``pretrained_config``, at ``self.dtype``:
        ``mm_encoder_class(pretrained_config.vision_config, self.dtype)``. Override it for a
        tower built another way. None builds no tower, and the model serves text only."""
        if self.mm_encoder_class is None:
            return None
        return self.mm_encoder_class(pretrained_config.vision_config, self.dtype)

    @property
    def language_model(self) -> nn.Module:
        return self.llm

    @property
    def dtype(self) -> torch.dtype:
        """The activation dtype. Set, not read off the first parameter, which may be a
        quantized one."""
        return self.model_config.torch_dtype

    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_size

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.dtype

    @property
    def text_embedding_layer(self) -> nn.Module:
        """The token embedding the fused path embeds text with."""
        return self.llm.model.embed_tokens

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
        """Admission check, run by the executor on every new request. The executor checks the
        token range of a decoder model only, so it is checked here against the language
        model's vocabulary; then, with a tower, `validate_media_request`."""
        num_embeddings = self.llm.lm_head.num_embeddings
        token_bound = request.py_prompt_token_bound
        if token_bound is not None:
            if token_bound >= num_embeddings:
                raise ValueError("Token ID out of range")
        elif not request.check_token_id_range(num_embeddings):
            raise ValueError("Token ID out of range")
        if self.mm_encoder is not None:
            self.validate_media_request(request)

    def validate_media_request(self, request) -> None:
        """Refuse, as its own error, a request whose media the tower cannot encode."""

    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        """Admission check: with no tower the model serves text only and refuses media or
        embeddings; with one it encodes its own media and refuses precomputed embeddings.
        M-RoPE positions alone pass."""
        if self.mm_encoder is None:
            if has_media_or_embeddings(mm_data):
                raise ValueError(
                    f"{type(self).__name__} builds no multimodal encoder (disable_mm_encoder) "
                    "and serves text only; send no images, videos or embeddings"
                )
        elif mm_data.get("multimodal_embedding") is not None:
            raise ValueError(
                f"{type(self).__name__} encodes its own media; precomputed multimodal "
                "embeddings are not supported"
            )

    def select_multimodal_params(self, multimodal_params, num_context_requests):
        """The mixin's selection, then only requests with media or their rows: an M-RoPE-only
        param (a disaggregated generation request, or a generation request's position
        deltas) has nothing to encode."""
        return [
            param
            for param in super().select_multimodal_params(multimodal_params, num_context_requests)
            if any(param.multimodal_data.get(key) is not None for key in _ENCODED_KEYS)
        ]

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        **kwargs,
    ):
        """The step's multimodal rows from the mixin, then the language model with every
        other argument the engine passed. The rows go in fused `inputs_embeds`, or with
        `language_model_takes_rows` as `multimodal_rows=(rows, mm_token_indices)` beside
        unchanged `input_ids`.

        The language model also takes what `get_language_model_extra_forward_kwargs` returns;
        the hook gets the same arguments as in the mixin's own forward. In the rows mode its
        `mm_inputs` holds `input_ids` and `inputs_embeds` as the language model gets them and
        the chunk's extra rows from `after_active_multimodal_embeddings` (one per multimodal
        row, not spread over the tokens), and `multimodal_rows` is among its keyword
        arguments."""
        raw_input_ids = input_ids
        multimodal_params = kwargs.pop("multimodal_params", [])
        prepare_kwargs = dict(
            input_ids=input_ids,
            positions=position_ids,
            multimodal_params=multimodal_params,
            num_context_requests=attn_metadata.num_contexts,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        if self.language_model_takes_rows:
            rows = self.prepare_multimodal_rows(**prepare_kwargs)
            # The extra rows are kept even when the chunk has no multimodal row, as fusion does.
            input_ids, active_embeddings, extra_embeds = rows or (input_ids, [], ())
            multimodal_rows = None
            if active_embeddings:
                embeds = (
                    active_embeddings[0]
                    if len(active_embeddings) == 1
                    else torch.cat(active_embeddings)
                )
                mm_token_indices = kwargs.get("mm_token_indices")
                assert (
                    mm_token_indices is not None and mm_token_indices.shape[0] == embeds.shape[0]
                ), (
                    f"the step has {embeds.shape[0]} multimodal rows and "
                    f"{None if mm_token_indices is None else mm_token_indices.shape[0]} "
                    "placeholder rows (mm_token_indices)"
                )
                multimodal_rows = (embeds, mm_token_indices)
            kwargs["multimodal_rows"] = multimodal_rows
            mm_inputs = PreparedLlmInputs(input_ids, inputs_embeds, extra_embeds)
        else:
            mm_inputs = self.prepare_multimodal_inputs(**prepare_kwargs)
            if inputs_embeds is not None:
                # As in the mixin's forward: the caller's embeddings replace input_ids.
                assert mm_inputs.inputs_embeds is None, (
                    "the engine passed inputs_embeds for a media step"
                )
                mm_inputs = PreparedLlmInputs(None, inputs_embeds, mm_inputs.extra_embeds)
        # As in the mixin's forward: the language model arguments and the target's extra ones
        # (for example DeepStack rows). They go over every argument the engine passed.
        llm_kwargs = self.get_language_model_forward_kwargs(
            attn_metadata=attn_metadata,
            input_ids=mm_inputs.input_ids,
            raw_input_ids=raw_input_ids,
            position_ids=position_ids,
            inputs_embeds=mm_inputs.inputs_embeds,
            mm_inputs=mm_inputs,
            return_context_logits=return_context_logits,
            multimodal_params=multimodal_params,
            num_generation_requests=attn_metadata.num_generations,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        return self.llm.forward(**{"spec_metadata": spec_metadata, **kwargs, **llm_kwargs})

    def load_weights(self, weights, weight_mapper=None) -> None:
        """The tower loads the keys under `mm_encoder_weight_prefix` and the language model
        every other key, with the engine's `weight_mapper` when its `load_weights` takes one;
        each checks its own coverage. The base renames no key. Without a tower the encoder keys
        are left unread.

        The engine binds its mapper to this model. The mapper is bound to the language model
        before it gets it, as in the Modeling V1 composite models, since the language model's
        keys name its own modules."""
        encoder_weights, text_weights = {}, {}
        for key in weights.keys():
            if not key.startswith(self.mm_encoder_weight_prefix):
                text_weights[key] = weights[key]
            elif self.mm_encoder is not None:
                encoder_weights[key] = weights[key]
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(encoder_weights)
        if (
            weight_mapper is not None
            and "weight_mapper" in inspect.getfullargspec(self.llm.load_weights).args
        ):
            weight_mapper.init_model_and_config(self.llm, self.llm.model_config)
        call_load_weights(self.llm.load_weights, text_weights, weight_mapper)
