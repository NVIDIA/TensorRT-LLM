# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 on Rubin with attention DP and four or sixteen expert-parallel ranks."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache

from . import weights as _weights
from ._model import Glm52Model, setup_aliases
from ._mtp import build_mtp_draft
from ._mtp_worker import RubinMTPEagleWorker
from .ops.gemm.low_m import install_low_m_methods


class ModelingV2Glm52Nvfp4Sm107HighThroughput(
    SpecDecOneEngineForCausalLM[Glm52Model, PretrainedConfig]
):
    """A GLM-owned decoder and MTP stack with DEP4/DEP16 checkpoint and KV layouts."""

    supports_nvfp4_kv_cache = True

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls, pretrained_config: PretrainedConfig | None = None
    ) -> Literal["V2"]:
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls, pretrained_config: PretrainedConfig | None = None
    ) -> Literal["PYTHON"]:
        return "PYTHON"

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        mapping = model_config.mapping
        if torch.cuda.get_device_capability() != (10, 7):
            raise ValueError("The GLM high-throughput target requires SM107")
        if not (
            mapping.world_size == mapping.tp_size == mapping.moe_ep_size
            and mapping.world_size in (4, 16)
            and mapping.moe_tp_size == 1
            and mapping.enable_attention_dp
            and mapping.pp_size == mapping.cp_size == 1
        ):
            raise ValueError(
                "The GLM high-throughput target requires attention DP with TP4/EP4 or TP16/EP16"
            )
        if not model_config.quant_config.quant_mode.has_fp4_kv_cache():
            raise ValueError("The GLM high-throughput target requires an NVFP4 KV cache")

        model_config = _weights.normalize_model_config(model_config)
        model = Glm52Model(model_config)
        DecoderModelForCausalLM.__init__(
            self,
            model,
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
        self.spec_config = model_config.spec_config
        self.draft_config = None
        self.draft_model = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        if self.spec_config is not None:
            if not (
                self.spec_config.spec_dec_mode.is_mtp_eagle_one_model()
                and not self.spec_config.uses_external_draft_model
                and not self.spec_config.uses_replacement_heads
            ):
                raise ValueError(
                    "The GLM high-throughput target requires the checkpoint's MTP layer"
                )
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            self.draft_model = build_mtp_draft(model_config, None, self.lm_head, self.model)
            self.epilogue.append(self.draft_model)
            self.spec_worker = RubinMTPEagleWorker(
                self.spec_config,
                model_config=model_config,
                mapping=mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
            )
            self.spec_worker.set_draft_model(self.draft_model)
            self.spec_worker.set_target_model(self)
            self.epilogue.append(self.spec_worker)
            self.num_hidden_layers = self.config.num_hidden_layers
            self.model.layers.extend(self.draft_model.mtp_layers)
        self.layer_idx = -1

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        install_low_m_methods(self)

    def load_weights(self, weights, *args, **kwargs) -> None:
        _weights.load_weights(self, weights)

    def setup_aliases(self) -> None:
        setup_aliases(self)
