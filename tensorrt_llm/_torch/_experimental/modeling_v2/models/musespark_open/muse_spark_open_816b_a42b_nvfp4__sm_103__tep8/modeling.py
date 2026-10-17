# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: muse-spark-open-816b-a42b-nvfp4 / sm_103 / tep8.

Muse Spark Open, NVFP4 export: 62 decoder layers, hidden 8192, 128 query and 16
key-value heads of width 64; every fourth layer counted from the last is a
full-attention NoPE layer and every other layer a 2048-token sliding-window RoPE
layer; the MoE routes each token to 8 of 256 experts at half the residual width.
``tep8`` is the attention heads tensor-parallel and the experts expert-parallel
over eight ranks. Speculation is DFlash, whose drafter is a separate checkpoint,
or none.

The shell is the built-in ``MuseSparkOpenOmniForConditionalGeneration``: the same
module tree, the same weights, the same engine preferences (KV cache manager,
transceiver, model defaults) and the same speculative-decoding path, because the
built-in ``MuseSparkOpenForCausalLM`` it wraps carries the one-engine DFlash
worker and builds the drafter from ``speculative_model``. Construction builds the
built-in model, then adopts its text model as ``MuseSparkOpenTep8ModelingV2``,
whose forward hands each step to ``PrefillTarget`` or ``DecodeTarget``.

Both targets call the built-in modules, so a step launches the kernels the
built-in implementation launches:

* ``PrefillTarget`` is the general case -- context rows, possibly with
  generation rows beside them -- and runs the built-in text-model forward.
* ``DecodeTarget`` is the specialization, routed to only when a step has no
  context rows: with DFlash, each request's last accepted token and its drafts.
  Its forward is the built-in forward's decode path written out over the
  built-in modules. The prefill sequence-parallel split, which needs context
  rows, is absent. A decode op that replaces a built-in call replaces it here.

Still built-in, and reached from here when a change needs it: the drafter and
the DFlash worker (``llm.draft_model``, ``llm.spec_worker``), the logits
processor and the weight loader (``weights.py``). Entries only this target calls
go under ``ops/<category>/`` beside this file.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

from . import weights as _weights


class MuseSparkOpenTep8ModelingV2(MuseSparkOpenModel):
    """The text model: the built-in ``MuseSparkOpenModel``, whose forward picks a target per step.

    Never constructed directly: ``adopt`` re-classes the built-in text model the
    shell has already built, so the module tree, the parameters and their names
    stay the built-in model's. The class adds methods and plain attributes only.
    """

    _prefill: "PrefillTarget"
    _decode: "DecodeTarget"

    @classmethod
    def adopt(cls, model: MuseSparkOpenModel) -> "MuseSparkOpenTep8ModelingV2":
        """Re-class the built-in text model as this core and build its two targets.

        Raises:
            TypeError: if ``model`` is not exactly the built-in ``MuseSparkOpenModel``.
        """
        if type(model) is not MuseSparkOpenModel:
            raise TypeError(
                f"{cls.__name__} adopts the built-in MuseSparkOpenModel, got {type(model).__name__}"
            )
        model.__class__ = cls
        model._prefill = PrefillTarget(model)
        model._decode = DecodeTarget(model)
        return model

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment. A mixed batch -- context and generation
        rows in one step -- goes to prefill, so prefill is the general case and
        decode the specialization that may assume it has no context rows.
        """
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them.

    Runs the built-in text-model forward, including its prefill sequence-parallel
    split when that is enabled.
    """

    def __init__(self, core: MuseSparkOpenTep8ModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return MuseSparkOpenModel.forward(
            self.core,
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    The built-in forward's decode path over the built-in modules: the token
    embedding, the weightless embedding norm and ``m_emb``, the FP32 residual
    stream, the decoder layers -- each given the next layer's pre-QKV norm so it
    can hand over that layer's attention input -- and the final norm. Every
    keyword the step carries (``spec_metadata`` among them, for the DFlash
    hidden-state capture) reaches every layer, as in the built-in forward.
    """

    def __init__(self, core: MuseSparkOpenTep8ModelingV2) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = core.embed_tokens(input_ids)
        embeds = core.embed_norm(inputs_embeds.to(core.dtype))
        if core.m_emb != 1.0:
            embeds = embeds * core.m_emb
        hidden_states = embeds.to(torch.float32)

        layers = core.layers
        precomputed_attn_input = None
        for index, decoder_layer in enumerate(layers):
            next_norm = layers[index + 1].pre_qkv_proj_norm if index + 1 < len(layers) else None
            result = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                precomputed_attn_input=precomputed_attn_input,
                next_layer_norm=next_norm,
                sp_num_tokens=None,
                **kwargs,
            )
            if next_norm is None:
                hidden_states = result
            else:
                hidden_states, precomputed_attn_input = result
        return core.norm(hidden_states.to(core.dtype))


class ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep8(MuseSparkOpenOmniForConditionalGeneration):
    """Muse-Spark-Open-816B-A42B-NVFP4 on sm_103 at tep8: the built-in model with this target's core."""

    def __init__(self, model_config: ModelConfig, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        # The built-in constructor returns early when it is re-entered; adopt once.
        if not isinstance(self.llm.model, MuseSparkOpenTep8ModelingV2):
            MuseSparkOpenTep8ModelingV2.adopt(self.llm.model)

    def load_weights(self, weights, weight_mapper=None):
        return _weights.load(self, weights, weight_mapper)
