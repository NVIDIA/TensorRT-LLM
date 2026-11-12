# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The speculative-decoding base of a Kimi K3 target.

A K3 target's causal-LM class derives from :class:`KimiK3SpecForCausalLM` in place of a
plain ``DecoderModelForCausalLM``. With a one-engine DSpark ``speculative_config`` whose
drafter is a standalone MLA checkpoint, it builds the family's drafter and worker itself:

* the drafter's own ``ModelConfig`` from its checkpoint, with the execution layout of the
  target engine it runs inside (every row of the target's mapping, the target's KV quant
  algo and ``extra_attrs``) -- the config the engine's weight loader and KV cache creator
  read off ``draft_config``;
* :class:`~.drafter.K3DSparkDrafter`, appended to the epilogue;
* :class:`~.worker.K3DSparkWorker`, bound to both models and appended to the epilogue.

Its forward is the one-engine step: the target, the target logits in the layout the worker's
acceptance reads, then the worker (acceptance, drafting).

Target constants, set on the target's class (no runtime switch reads them):

* ``dspark_path``: the drafter and worker's :class:`~.drafter.K3DSparkPath`.
* ``dspark_context_kv_only``: the target is a disaggregated context server; its drafter
  stores the context latent only (see ``K3DSparkDrafter``).
* ``spec_decode_prologue``: the speculative decode step's head kernel the engine runs as the
  ``decode_prologue`` runtime hook, or None.
"""

import inspect
from dataclasses import replace
from typing import Callable, Generic, Optional, TypeVar

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig, TConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.speculative import SpecMetadata, should_use_separate_draft_kv_cache
from tensorrt_llm.quantization.mode import QuantAlgo

from .drafter import K3DSparkPath, build_k3_dspark_drafter
from .worker import K3DSparkWorker

# The target's decoder model (a ``DecoderModel``).
TModel = TypeVar("TModel", bound=nn.Module)


def _draft_config_kwargs(model_config: ModelConfig, spec_config) -> dict:
    """``ModelConfig.from_pretrained`` kwargs of the standalone drafter's config.

    The drafter is a separate checkpoint, so it gets its own ``ModelConfig``; these are the
    execution-layout properties it inherits from the target engine it runs inside.
    """
    requested_moe_backend = getattr(spec_config, "moe_backend", None)
    return dict(
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=(
            model_config.moe_backend if requested_moe_backend is None else requested_moe_backend
        ),
        # The drafter runs on every row; keep it TP where the TP group shares requests (helix
        # + attention DP at tp_size 1).
        mapping=model_config.mapping.mapping_for_full_rows(),
        spec_config=None,
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
        use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
        # The user's value, not the engine's: the executor raises it past this and never
        # writes back, so the drafter reads ``_runtime_position_ceiling``. None sizes position
        # tables from max_position_embeddings.
        max_seq_len=model_config.max_seq_len,
    )


def _inherit_draft_kv_cache_quant_algo(draft_config: ModelConfig, target_config: ModelConfig):
    """The target's KV quant algo, unless the target requests a draft override."""
    algo = target_config.quant_config.kv_cache_quant_algo
    override = target_config.extra_attrs.get("draft_kv_cache_quant_algo_override")
    if override is not None:
        algo = QuantAlgo(override)
    draft_config.quant_config.kv_cache_quant_algo = algo
    if override is not None and draft_config.quant_config_dict is not None:
        for layer_quant_config in draft_config.quant_config_dict.values():
            layer_quant_config.kv_cache_quant_algo = algo


def uses_k3_dspark(spec_config) -> bool:
    """Whether ``spec_config`` is one-engine DSpark with a separately shipped drafter."""
    return (
        spec_config is not None
        and spec_config.spec_dec_mode.use_one_engine()
        and spec_config.spec_dec_mode.is_dspark()
        and not bool(getattr(spec_config, "draft_is_embedded_in_target", False))
    )


class KimiK3SpecForCausalLM(DecoderModelForCausalLM[TModel, TConfig], Generic[TModel, TConfig]):
    """``DecoderModelForCausalLM`` with the family's one-engine DSpark drafter and worker."""

    dspark_path: K3DSparkPath = K3DSparkPath()
    dspark_context_kv_only: bool = False
    spec_decode_prologue: Optional[Callable] = None

    # The engine's one-engine fast paths (vocab-sharded key exchange, overlap loop on the
    # execution stream) are not taken; the worker's vocab-parallel head serves the selection.
    spec_decode_fast_paths: bool = False

    @property
    def runtime_hooks(self) -> Optional[RuntimeHooks]:
        """The speculative decode step's head in one kernel, where the target declares one."""
        if self.spec_decode_prologue is None:
            return None
        return replace(DEFAULT_RUNTIME_HOOKS, decode_prologue=self.spec_decode_prologue)

    def __init__(
        self,
        model: TModel,
        model_config: ModelConfig[TConfig],
        hidden_size: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ) -> None:
        """Build the target, then the drafter and the worker.

        Args:
            model: the target's ``DecoderModel``.
            model_config: the target's ``ModelConfig``. Under helix it carries the repurposed
                (CP -> TP) mapping the drafter is built under, and the CP original as
                ``_helix_mapping_with_cp``.
            hidden_size, vocab_size: the text config's, for composite configs that do not
                expose them at the top level.

        Raises:
            ValueError: a ``speculative_config`` other than one-engine DSpark with a standalone
                drafter.
        """
        if hidden_size is None:
            hidden_size = model_config.pretrained_config.hidden_size
        if vocab_size is None:
            vocab_size = model_config.pretrained_config.vocab_size
        super().__init__(model, config=model_config, hidden_size=hidden_size, vocab_size=vocab_size)
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        spec_config = getattr(model_config, "spec_config", None)
        self.spec_config = spec_config
        if spec_config is not None:
            if not uses_k3_dspark(spec_config):
                raise ValueError(
                    "Kimi K3 speculative decoding is one-engine DSpark with a standalone "
                    f"drafter checkpoint; got {spec_config.spec_dec_mode}."
                )
            self._build_dspark(model_config, spec_config)
        # The final hidden state is captured under layer id -1.
        self.layer_idx = -1

    def _build_dspark(self, model_config: ModelConfig, spec_config) -> None:
        draft_config = ModelConfig.from_pretrained(
            spec_config.speculative_model, **_draft_config_kwargs(model_config, spec_config)
        )
        _inherit_draft_kv_cache_quant_algo(draft_config, model_config)
        # Shared with the target: the KV cache creator and the drafter read the engine's
        # entries (position headroom, ...) off either config.
        draft_config.extra_attrs = model_config.extra_attrs
        self.draft_config = draft_config
        self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(spec_config)

        draft_model = build_k3_dspark_drafter(
            draft_config,
            spec_config,
            path=self.dspark_path,
            helix_mapping=getattr(model_config, "_helix_mapping_with_cp", None),
            context_kv_only=self.dspark_context_kv_only,
        )
        if spec_config.spec_dec_mode.is_parallel_draft():
            draft_model.logits_processor = self.logits_processor
        self.draft_model = draft_model
        self.epilogue.append(draft_model)

        worker = K3DSparkWorker(
            spec_config,
            model_config.mapping.mapping_for_full_rows(),
            self.use_separate_draft_kv_cache,
            path=self.dspark_path,
        )
        # Only this processor makes the target logits ``lm_head(hidden_states)``, which the
        # worker's vocab-parallel selection computes per shard.
        worker.plain_logits_processor_type = type(self.logits_processor)
        worker.set_draft_model(draft_model)
        worker.set_target_model(self)
        self.spec_worker = worker
        self.epilogue.append(worker)

    def load_draft_weights(self, weights, weight_mapper=None) -> None:
        """Load the drafter checkpoint, then share the target's LM head with the drafter."""
        if "weight_mapper" in inspect.getfullargspec(self.draft_model.load_weights).args:
            self.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)
        else:
            self.draft_model.load_weights(weights=weights)
        self.draft_model.load_weights_from_target_model(self)

    def set_guided_decoder(self, guided_decoder) -> bool:
        if self.spec_worker is not None:
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]

        if self.spec_worker is None:
            return self.logits_processor.forward(
                hidden_states, self.lm_head, attn_metadata, return_context_logits
            )

        # The target logits, in the layout the worker's acceptance reads.
        logits = self.spec_worker.target_logits(
            self.spec_worker.target_head_rows(hidden_states, attn_metadata, spec_metadata),
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        # Multimodal wrappers replace input_ids with fused inputs_embeds and forward the
        # pre-fusion token ids as ``orig_input_ids``.
        spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
        spec_position_ids = position_ids
        if attn_metadata.padded_num_tokens is not None:
            if spec_input_ids is not None:
                spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
            if position_ids is not None:
                spec_position_ids = position_ids[..., : attn_metadata.num_tokens]
        return self.spec_worker(
            input_ids=spec_input_ids,
            position_ids=spec_position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )
