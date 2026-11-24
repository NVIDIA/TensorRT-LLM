# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned text and draft module construction with the engine's execution interface."""

from typing import Generic

from tensorrt_llm._torch.model_config import ModelConfig, TConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    _SPECULATIVE_POSITION_HEADROOM,
    _get_requested_draft_moe_backend,
    _set_draft_kv_cache_quant_algo,
)
from tensorrt_llm._torch.models.modeling_speculative import (
    SpecDecOneEngineForCausalLM as EngineCausalLM,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, TModel
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache


class TargetCausalLM(EngineCausalLM[TModel, TConfig], Generic[TModel, TConfig]):
    def __init__(self, model, model_config, hidden_size=None, vocab_size=None):
        cfg = model_config.pretrained_config
        self._initialize_decoder(
            model,
            config=model_config,
            hidden_size=hidden_size if hidden_size is not None else cfg.hidden_size,
            vocab_size=vocab_size if vocab_size is not None else cfg.vocab_size,
        )
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        self.spec_config = getattr(model_config, "spec_config", None)
        spec = self.spec_config
        if spec is not None:
            if not spec.spec_dec_mode.is_dflash():
                raise ValueError("The NVFP4 target supports only the DFlash2 MLA drafter")
            self.draft_config = ModelConfig.from_pretrained(
                spec.speculative_model,
                trust_remote_code=True,
                attn_backend=model_config.attn_backend,
                moe_backend=_get_requested_draft_moe_backend(model_config, spec),
                mapping=model_config.mapping.mapping_for_full_rows(),
                spec_config=None,
                max_num_tokens=model_config.max_num_tokens,
                moe_max_num_tokens=model_config.moe_max_num_tokens,
            )
            _set_draft_kv_cache_quant_algo(self.draft_config, model_config)
            self.draft_config.extra_attrs = model_config.extra_attrs
            self.draft_config.extra_attrs[_SPECULATIVE_POSITION_HEADROOM] = (
                2 * spec.tokens_per_gen_step
            )
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(spec)
        self.layer_idx = -1

    def _initialize_decoder(self, model, *, config, hidden_size, vocab_size):
        DecoderModelForCausalLM.__init__(
            self, model, config=config, hidden_size=hidden_size, vocab_size=vocab_size
        )
