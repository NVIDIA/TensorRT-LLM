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
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM, DeepseekV3Model
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.speculative import SpecMetadata

from . import weights as _weights
from .ops.embedding.fused_embedding import fused_embedding
from .ops.gemm.dense_mlp_small_m import INTER_PER_RANK, dense_mlp_small_m

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

    The step is written out layer by layer for this deployment's configuration,
    which ``__init__`` checks once:

    - Layers 0-2 (dense, unquantized): the attention all-reduces its own output,
      the residual add and ``post_attention_layernorm`` follow, and the MLP's
      all-reduce carries the residual add and the next layer's input norm.
    - Layers 3-77 (MoE): both all-reduces carry the residual add and the
      following norm; the experts finalize before theirs (16 ranks span nodes).
    - The last all-reduce applies the model's final norm (``next_layer_layernorm``).

    The embedding table is whole on every rank, so the lookup is one local
    kernel. The dense MLP runs as two weight-streaming kernels on this rank's
    768 intermediate columns for up to 8 rows (every c1 step: one request and
    its 7 drafts); a larger batch takes the generic gated MLP.
    """

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core
        for layer in core.layers[: core.num_hidden_layers]:
            assert layer.allreduce is not None and layer.next_layer_layernorm is not None
            assert layer.input_layernorm.nvfp4_scale is None, "kv_a_proj is not NVFP4"
            if isinstance(layer.mlp, GatedMLP):
                assert (
                    layer.mlp_tp_size == _TP
                    and not layer.disable_attn_allreduce
                    and not layer.fusion_config.PRE_MLP_FUSION
                    and layer.fusion_config.POST_MLP_FUSION
                    and layer.post_attention_layernorm.nvfp4_scale is None
                ), f"layer {layer.layer_idx}: unexpected dense-layer configuration"
            else:
                assert (
                    layer.disable_attn_allreduce
                    and layer.fusion_config.PRE_MOE_FUSION
                    and layer.fusion_config.POST_MOE_FUSION
                    and layer.mapping.is_multi_node()
                ), f"layer {layer.layer_idx}: unexpected MoE-layer configuration"

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        if inputs_embeds is None:
            inputs_embeds = fused_embedding(input_ids, core.embed_tokens.weight)
        residual = inputs_embeds
        hidden_states = core.layers[0].input_layernorm(inputs_embeds)
        small_m = torch.ops.trtllm.glm_dense_mlp_small_m_supported(
            hidden_states.shape[0], core.config.hidden_size, INTER_PER_RANK
        )
        for layer in core.layers[: core.num_hidden_layers]:
            if isinstance(layer.mlp, GatedMLP):
                hidden_states, residual = self._dense_layer(
                    layer, position_ids, hidden_states, residual, attn_metadata, small_m
                )
            else:
                hidden_states, residual = self._moe_layer(
                    layer, position_ids, hidden_states, residual, attn_metadata
                )
        return hidden_states

    @staticmethod
    def _dense_layer(layer, position_ids, hidden_states, residual, attn_metadata, small_m):
        hidden_states = layer.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=True),
        )
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        if small_m:
            # The rank partial; the all-reduce below sums it over the 16 ranks.
            hidden_states = dense_mlp_small_m(
                hidden_states, layer.mlp.gate_up_proj.weight, layer.mlp.down_proj.weight
            )
        else:
            hidden_states = layer.mlp(
                hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
        return layer.allreduce(
            hidden_states,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=layer.next_layer_layernorm.weight,
                eps=layer.next_layer_layernorm.variance_epsilon,
            ),
        )

    @staticmethod
    def _moe_layer(layer, position_ids, hidden_states, residual, attn_metadata):
        hidden_states = layer.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
        )
        hidden_states, residual = layer.allreduce(
            hidden_states,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=layer.post_attention_layernorm.weight,
                eps=layer.post_attention_layernorm.variance_epsilon,
                trigger_completion_at_end=False,
            ),
        )
        hidden_states = layer.mlp(
            hidden_states,
            None,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(enable_allreduce=False),
            do_finalize=True,
        )
        return layer.allreduce(
            hidden_states,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=layer.next_layer_layernorm.weight,
                eps=layer.next_layer_layernorm.variance_epsilon,
                trigger_completion_at_end=False,
            ),
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
