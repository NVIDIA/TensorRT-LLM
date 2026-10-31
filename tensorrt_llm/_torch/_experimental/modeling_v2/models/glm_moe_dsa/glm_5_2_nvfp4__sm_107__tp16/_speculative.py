# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-engine speculative model construction for the GLM-5.2 target."""

from typing import Generic

from tensorrt_llm._torch.model_config import ModelConfig, TConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    _SPECULATIVE_POSITION_HEADROOM,
    SpecDecOneEngineForCausalLM,
    _copy_model_config_with_moe_backend,
    _get_requested_draft_moe_backend,
    _replacement_mtp_model_config,
    _set_draft_kv_cache_quant_algo,
    external_drafter_config_kwargs,
    get_draft_model,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, TModel
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache

from ._mtp import build_mtp_draft


class Glm52SpecDecOneEngineForCausalLM(
    SpecDecOneEngineForCausalLM[TModel, TConfig], Generic[TModel, TConfig]
):
    """Generic one-engine execution with GLM-owned embedded MTP construction."""

    def __init__(
        self,
        model: TModel,
        model_config: ModelConfig[TConfig],
        hidden_size: int | None = None,
        vocab_size: int | None = None,
    ) -> None:
        # Composite configs (e.g. vision-language wrappers) may not expose
        # hidden_size/vocab_size at the top level; callers can pass the
        # text-config values explicitly.
        if hidden_size is None:
            hidden_size = model_config.pretrained_config.hidden_size
        if vocab_size is None:
            vocab_size = model_config.pretrained_config.vocab_size
        DecoderModelForCausalLM.__init__(
            self, model, config=model_config, hidden_size=hidden_size, vocab_size=vocab_size
        )
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        spec_config = getattr(model_config, "spec_config", None)
        self.spec_config = spec_config
        if spec_config and spec_config.spec_dec_mode.use_one_engine():
            # Only create draft_model for modes MTP, Eagle3 (not SA)
            if not spec_config.spec_dec_mode.is_sa():
                requested_draft_moe_backend = _get_requested_draft_moe_backend(
                    model_config, spec_config
                )
                if spec_config.spec_dec_mode.is_eagle3_one_model():
                    if spec_config.eagle3_model_arch == "mistral_large3":
                        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import (
                            MistralConfigLoader,
                        )

                        self.draft_config = MistralConfigLoader().load(
                            spec_config.speculative_model,
                            mapping=model_config.mapping,
                            moe_backend=requested_draft_moe_backend,
                            moe_max_num_tokens=model_config.moe_max_num_tokens,
                            max_num_tokens=model_config.max_num_tokens,
                            moe_load_balancer=model_config.moe_load_balancer,
                            skip_create_weights_in_init=True,
                        )
                        if getattr(spec_config, "moe_backend", None) is not None:
                            # Unlike ModelConfig.from_pretrained, the Mistral
                            # loader does not resolve AUTO after loading quant
                            # metadata. Resolve it against the draft config now,
                            # before constructing any draft modules.
                            self.draft_config = _copy_model_config_with_moe_backend(
                                self.draft_config, requested_draft_moe_backend
                            )
                    elif spec_config.eagle3_model_arch == "llama3":
                        self.draft_config = ModelConfig.from_pretrained(
                            model_config.spec_config.speculative_model,
                            trust_remote_code=True,
                            attn_backend=model_config.attn_backend,
                            moe_backend=requested_draft_moe_backend,
                            mapping=model_config.mapping,
                            spec_config=model_config.spec_config,
                            max_num_tokens=model_config.max_num_tokens,
                            moe_max_num_tokens=model_config.moe_max_num_tokens,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported eagle3 model architecture for draft model: {spec_config.eagle3_model_arch}"
                        )
                    _set_draft_kv_cache_quant_algo(self.draft_config, model_config)
                    self.draft_config.extra_attrs = model_config.extra_attrs

                elif spec_config.uses_external_draft_model:
                    self.draft_config = ModelConfig.from_pretrained(
                        spec_config.speculative_model,
                        trust_remote_code=True,
                        attn_backend=model_config.attn_backend,
                        moe_backend=requested_draft_moe_backend,
                        mapping=model_config.mapping,
                        spec_config=None,
                        max_num_tokens=model_config.max_num_tokens,
                        moe_max_num_tokens=model_config.moe_max_num_tokens,
                    )
                    _set_draft_kv_cache_quant_algo(self.draft_config, model_config)
                    self.draft_config.extra_attrs = model_config.extra_attrs
                    self.draft_config.extra_attrs[_SPECULATIVE_POSITION_HEADROOM] = (
                        2 * spec_config.tokens_per_gen_step
                    )

                elif spec_config.spec_dec_mode.is_external_drafter():
                    self.draft_config = ModelConfig.from_pretrained(
                        model_config.spec_config.speculative_model,
                        **external_drafter_config_kwargs(model_config, spec_config),
                    )
                    _set_draft_kv_cache_quant_algo(self.draft_config, model_config)
                    self.draft_config.extra_attrs = model_config.extra_attrs

                elif spec_config.uses_replacement_heads:
                    self.draft_config = _replacement_mtp_model_config(model_config)

                self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(spec_config)

                if (
                    spec_config.spec_dec_mode.is_mtp_one_model()
                    and not spec_config.uses_external_draft_model
                ):
                    self.draft_model = build_mtp_draft(
                        model_config, self.draft_config, self.lm_head, self.model
                    )
                else:
                    self.draft_model = get_draft_model(
                        model_config, self.draft_config, self.lm_head, self.model
                    )
                if spec_config.uses_replacement_heads:
                    self.draft_config = self.draft_model.model_config
                if self.draft_model is not None:
                    self.epilogue.append(self.draft_model)
                if (spec_config.spec_dec_mode.is_parallel_draft()) and self.draft_model is not None:
                    self.draft_model.logits_processor = self.logits_processor

            # spec_worker is created for all one-engine modes (MTP, Eagle3, SA)
            self.spec_worker = get_spec_worker(
                model_config.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
            )
            if self.spec_worker is not None:
                # Cache the static draft->target vocab map now that the draft
                # model is loaded, so workers read self._d2t instead of probing
                # draft_model.model.d2t on every forward.
                self.spec_worker.set_draft_model(self.draft_model)
                self.epilogue.append(self.spec_worker)
        self.layer_idx = -1
