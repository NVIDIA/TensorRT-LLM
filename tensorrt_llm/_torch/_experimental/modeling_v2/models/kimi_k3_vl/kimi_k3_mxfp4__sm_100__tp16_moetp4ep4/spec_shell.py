# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-engine speculative decoding for this target's causal LM (`modeling.py`): SA through the speculative runtime's
worker, DSpark with a standalone drafter checkpoint through this target's drafters (`drafter.py` GQA, `drafter_mla.py`
MLA) and worker (`spec_worker.py`).

Functions of the model, which the target class calls from its own ``__init__``, ``forward``, ``load_draft_weights``
and ``set_guided_decoder``: the parts of the built-in ``SpecDecOneEngineForCausalLM`` these modes run, with the
draft-config helpers of ``_torch/models/modeling_speculative.py`` in this file and the family's ``drafter_config``
(``glm_5_2_nvfp4__sm_107__low_latency/_speculative.py`` does the same for MTP). A mode, or a drafter, this target does
not build raises ``NoTarget``: under ``modeling_v2="auto"`` the engine then runs Modeling V1.
"""

from __future__ import annotations

import inspect
from typing import Dict, Optional

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    drafter_config as _drafter_config,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.drafter_config import (
    external_drafter_config_kwargs,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    get_spec_worker,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm.quantization.mode import QuantAlgo

from . import spec_worker as _spec_worker
from .drafter import K3DSparkDrafter
from .drafter_mla import K3MLADSparkDrafter


def set_draft_kv_cache_quant_algo(draft_config: ModelConfig, target_config: ModelConfig) -> None:
    """The draft KV cache takes the target's KV dtype unless the target requests a draft override (the built-in
    ``modeling_speculative._set_draft_kv_cache_quant_algo``)."""
    algo = target_config.quant_config.kv_cache_quant_algo
    override = target_config.extra_attrs.get("draft_kv_cache_quant_algo_override")
    if override is not None:
        algo = QuantAlgo(override)
    draft_config.quant_config.kv_cache_quant_algo = algo
    if override is not None and draft_config.quant_config_dict is not None:
        for layer_quant_config in draft_config.quant_config_dict.values():
            layer_quant_config.kv_cache_quant_algo = algo


def slice_spec_position_ids(
    position_ids: Optional[torch.Tensor], num_tokens: int
) -> Optional[torch.Tensor]:
    """Speculative position IDs sliced along the token dimension."""
    if position_ids is None:
        return None
    return position_ids[..., :num_tokens]


def draft_model_class(draft_config: ModelConfig):
    """The drafter class of a standalone DSpark checkpoint: `K3MLADSparkDrafter` for an MLA-shaped drafter
    (Inferact-Kimi-K3-DSpark), else `K3DSparkDrafter` for the Qwen3-shaped GQA drafter (RadixArk-Kimi-K3-DSpark).
    The built-in DSpark drafter dispatches on the same test (``is_mla``).

    Raises:
        NoTarget: a GQA drafter `K3DSparkDrafter` does not build.
    """
    cfg = draft_config.pretrained_config
    if _drafter_config.is_mla(cfg):
        return K3MLADSparkDrafter
    reason = _drafter_config.gqa_declines(cfg)
    if reason is not None:
        raise NoTarget(f"the Kimi K3 DSpark drafter does not build this drafter: {reason}")
    return K3DSparkDrafter


def build(model: nn.Module, model_config: ModelConfig) -> None:
    """Build ``model``'s speculative decoding from the text model's ``spec_config``, after
    ``DecoderModelForCausalLM.__init__`` and with ``model``'s speculative fields at their off values:

    * no speculation: nothing;
    * SA: the speculative runtime's worker (``get_spec_worker``), no drafter;
    * DSpark, standalone drafter (GQA or MLA): the drafter config from the draft checkpoint, the drafter
      (`draft_model_class`) sharing ``model``'s logits processor, and this target's ``KimiK3DSparkWorker`` with the
      drafter set on it. The K3 worker is not handed the target (``set_target_model``), so it keeps the stock
      worker's gathered target logits.

    The drafter and the worker are submodules of ``model``, so the model loader's post-load walk reaches them; both go
    on the epilogue.

    Raises:
        NoTarget: any other mode (DFlash, embedded DSpark, ...), or a drafter this target does not build.
    """
    spec_config = getattr(model_config, "spec_config", None)
    model.spec_config = spec_config
    if spec_config is None:
        return
    mode = spec_config.spec_dec_mode
    if mode.is_sa():
        model.spec_worker = get_spec_worker(
            spec_config,
            model_config,
            model_config.mapping.mapping_for_full_rows(),
            use_separate_draft_kv_cache=False,
            key_exchange=None,
            spec_decode_fast_paths=False,
        )
        model.spec_worker.set_draft_model(None)
        model.spec_worker.set_target_model(model)
        model.epilogue.append(model.spec_worker)
        return
    if not mode.is_dspark() or spec_config.draft_is_embedded_in_target:
        raise NoTarget(
            "the Kimi K3 target runs speculative decoding with SA or with a standalone DSpark drafter; "
            f"spec_dec_mode is {mode.name}"
            + (" with the draft embedded in the target" if mode.is_dspark() else "")
        )
    draft_config = ModelConfig.from_pretrained(
        spec_config.speculative_model,
        **external_drafter_config_kwargs(model_config, spec_config),
    )
    set_draft_kv_cache_quant_algo(draft_config, model_config)
    draft_config.extra_attrs = model_config.extra_attrs
    drafter_cls = draft_model_class(draft_config)
    model.draft_config = draft_config
    model.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(spec_config)
    # The drafter is built without the speculative config's mask id: the worker reads that id itself.
    model.draft_model = drafter_cls(
        draft_config, dflash_attention_backend=spec_config.attention_backend
    )
    model.epilogue.append(model.draft_model)
    model.draft_model.logits_processor = model.logits_processor
    worker = _spec_worker.KimiK3DSparkWorker(
        spec_config, model_config.mapping, model.use_separate_draft_kv_cache
    )
    worker.set_draft_model(model.draft_model)
    model.spec_worker = worker
    model.epilogue.append(worker)


def forward(
    model: nn.Module,
    attn_metadata: AttentionMetadata,
    input_ids: Optional[torch.IntTensor] = None,
    position_ids: Optional[torch.IntTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    spec_metadata: Optional[SpecMetadata] = None,
    resource_manager=None,
    **kwargs,
) -> torch.Tensor:
    """``model``'s step with its speculative worker: the text model (the capture layer -1's hidden states into
    ``spec_metadata``), the padded rows dropped, the target logits in the layout the worker's acceptance reads, then the
    worker (the accepted tokens and the next draft tokens)."""
    hidden_states = model.model(
        input_ids=input_ids,
        attn_metadata=attn_metadata,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        spec_metadata=spec_metadata,
        **kwargs,
    )
    if spec_metadata is not None and spec_metadata.is_layer_capture(model.layer_idx):
        spec_metadata.maybe_capture_hidden_states(model.layer_idx, hidden_states)
    if attn_metadata.padded_num_tokens is not None:
        hidden_states = hidden_states[: attn_metadata.num_tokens]
    logits = model.spec_worker.target_logits(
        hidden_states[spec_metadata.gather_ids],
        model.lm_head,
        model.logits_processor,
        attn_metadata,
        spec_metadata,
        model.draft_model,
    )
    spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
    spec_position_ids = position_ids
    if attn_metadata.padded_num_tokens is not None:
        if spec_input_ids is not None:
            spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
        if position_ids is not None:
            spec_position_ids = slice_spec_position_ids(position_ids, attn_metadata.num_tokens)
    return model.spec_worker(
        input_ids=spec_input_ids,
        position_ids=spec_position_ids,
        hidden_states=hidden_states,
        logits=logits,
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
        draft_model=model.draft_model,
        resource_manager=resource_manager,
    )


def load_draft_weights(model: nn.Module, weights: Dict, weight_mapper=None) -> None:
    """The drafter's checkpoint into ``model``'s drafter, then ``model``'s embedding and LM head shared with it."""
    if model.draft_model is None:
        raise ValueError("this Kimi K3 model has no drafter to load draft weights into")
    if "weight_mapper" in inspect.getfullargspec(model.draft_model.load_weights).args:
        model.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)
    else:
        model.draft_model.load_weights(weights=weights)
    model.draft_model.load_weights_from_target_model(model)


def set_guided_decoder(model: nn.Module, guided_decoder) -> bool:
    """Hand the guided decoder to ``model``'s speculative worker; False without one that takes it."""
    if hasattr(model.spec_worker, "set_guided_decoder"):
        return model.spec_worker.set_guided_decoder(guided_decoder)
    return False
