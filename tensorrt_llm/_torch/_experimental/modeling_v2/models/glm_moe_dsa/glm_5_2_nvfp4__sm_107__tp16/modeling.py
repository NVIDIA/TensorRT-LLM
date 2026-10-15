# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 on 16 Rubin GPUs (sm107), tp16: model construction and forward.

This target serves the generation side of a disaggregated deployment. Its
context server runs the built-in GLM-5.2 implementation (modeling_deepseekv3)
and hands over the MLA KV cache and the DSA indexer K cache it wrote, so this
target builds the same modules and loads the same weights: the caches it reads
are laid out exactly as they were written. Forwards that carry context rows
(engine warm-up, an aggregated deployment) run that same decoder stack in
``PrefillTarget``. Generation-only forwards -- every serving step of the
generation server -- run ``DecodeTarget``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM, DeepseekV3Model
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.speculative import SpecMetadata

from . import weights as _weights

# The deployment this target is written for; route() selects it on the same
# facts, and the constructor states them again for a direct construction.
_COMPUTE_CAPABILITY = (10, 7)
_TP = 16


class PrefillTarget:
    """Forwards that carry context rows: the decoder stack the context server runs."""

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return DeepseekV3Model.forward(
            self.core,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class DecodeTarget:
    """Generation-only forwards: the serving step of the generation server.

    Routed to only when the batch has no context rows, which the core decides
    per step; nothing in this class reads the phase back.
    """

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return DeepseekV3Model.forward(
            self.core,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class Glm52Nvfp4Sm107Tp16Core(DeepseekV3Model):
    """The decoder stack, with the per-step choice between the two targets."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    def post_load_weights(self) -> None:
        """Build the targets once the weights are final. Reached more than once
        (the shell and the loader's module walk), so it only builds once."""
        if self._decode is not None:
            return
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata: AttentionMetadata) -> "PrefillTarget | DecodeTarget":
        if self._decode is None:
            self.post_load_weights()
        return self._prefill if attn_metadata.num_contexts > 0 else self._decode

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class ModelingV2Glm52Nvfp4Sm107Tp16(
    SpecDecOneEngineForCausalLM[Glm52Nvfp4Sm107Tp16Core, PretrainedConfig]
):
    """GLM-5.2-NVFP4, sm107, tp16."""

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls, pretrained_config: Any = None
    ) -> Literal["V2"]:
        """The KV cache manager the context server's built-in implementation prefers."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls, pretrained_config: Any = None
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        """The transceiver runtime for the masked DSA indexer K-cache transfer."""
        return "PYTHON"

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        mapping = model_config.mapping
        assert torch.cuda.get_device_capability() == _COMPUTE_CAPABILITY, (
            f"this target runs on sm107, not {torch.cuda.get_device_capability()}"
        )
        assert (
            mapping.world_size == _TP
            and mapping.tp_size == _TP
            and mapping.moe_ep_size == 1
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
        ), f"this target is tp16 without attention DP or expert parallelism, not {mapping}"
        model_config = _weights.normalize_model_config(model_config)
        super().__init__(model=Glm52Nvfp4Sm107Tp16Core(model_config), model_config=model_config)
        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
        ):
            self.num_hidden_layers = self.config.num_hidden_layers
            self.model.layers.extend(self.draft_model.mtp_layers)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            spec_metadata=spec_metadata,
            resource_manager=resource_manager,
            **kwargs,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.load_weights(self, weights)

    def setup_aliases(self) -> None:
        DeepseekV3ForCausalLM.setup_aliases(self)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
