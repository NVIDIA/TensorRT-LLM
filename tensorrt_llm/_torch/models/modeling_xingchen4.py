# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XingChen4 TRT-LLM PyTorch model with mHC residual mixing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict

from ..attention_backend import AttentionMetadata
from ..distributed import AllReduce, AllReduceParams, MoEAllReduce
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.mhc.hyper_connection import mHC
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType
from .modeling_deepseekv3 import DeepseekV3Attention, Deepseekv3MoE, DeepseekV3WeightLoader
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, EagerFusionConfig, register_auto_model


def _patch_rope_scaling(config: PretrainedConfig) -> None:
    """Normalize XingChen4 rope scaling for TRT-LLM."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return
    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    if rope_type == "rope":
        rope_scaling = dict(rope_scaling)
        rope_scaling["type"] = "yarn"
        config.rope_scaling = rope_scaling


class XingChen4DecoderLayer(DecoderLayer):
    """Decoder layer with per-sub-block mHC."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.layer_idx = layer_idx
        self.config = model_config.pretrained_config
        config = self.config

        self.hidden_size = config.hidden_size
        self.n_streams = getattr(config, "hc_mult", getattr(config, "num_residual_streams", 1))
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.mapping = model_config.mapping
        mapping = self.mapping
        self.enable_attention_dp = mapping.enable_attention_dp
        self.tp_size = mapping.tp_size

        needs_tp_reduce = not self.enable_attention_dp and self.tp_size > 1
        self.self_attn = DeepseekV3Attention(
            model_config,
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            reduce_output=needs_tp_reduce,
        )

        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )

        if self.is_moe:
            self.mlp = Deepseekv3MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                shared_expert_intermediate_size=self.moe_intermediate_size
                * self.num_shared_experts,
                dtype=config.torch_dtype,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                layer_idx=layer_idx,
            )
        else:
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                overridden_tp_size=None,
                reduce_output=self.tp_size > 1,
                use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
            )

        self.fusion_config = EagerFusionConfig()

        self.allreduce = None
        self.moe_allreduce = None
        if not self.enable_attention_dp and self.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=mapping,
                strategy=model_config.allreduce_strategy,
                dtype=config.torch_dtype,
            )
            self.moe_allreduce = MoEAllReduce(mapping)
        self.disable_attn_allreduce = not needs_tp_reduce

        hc_eps = float(getattr(config, "hc_eps", 1e-6))
        sinkhorn_iters = int(
            getattr(config, "hc_sinkhorn_iters", getattr(config, "mhc_sinkhorn_iterations", 20))
        )
        mhc_norm_eps = float(getattr(config, "mhc_norm_eps", hc_eps))
        mhc_pre_eps = float(getattr(config, "mhc_pre_eps", hc_eps))
        mhc_sinkhorn_eps = float(getattr(config, "mhc_sinkhorn_eps", hc_eps))
        mhc_post_mult_value = float(getattr(config, "mhc_post_mult_value", 2.0))
        self.mhc_h_res_clamp_min = getattr(config, "mhc_h_res_clamp_min", None)
        self.mhc_h_res_clamp_max = getattr(config, "mhc_h_res_clamp_max", None)

        self.attn_hc = mHC(
            mult=self.n_streams,
            hidden_size=self.hidden_size,
            sinkhorn_iters=sinkhorn_iters,
            dtype=config.torch_dtype,
            eps=mhc_pre_eps,
            norm_eps=mhc_norm_eps,
            sinkhorn_eps=mhc_sinkhorn_eps,
            post_mult_value=mhc_post_mult_value,
        )
        self.ffn_hc = mHC(
            mult=self.n_streams,
            hidden_size=self.hidden_size,
            sinkhorn_iters=sinkhorn_iters,
            dtype=config.torch_dtype,
            eps=mhc_pre_eps,
            norm_eps=mhc_norm_eps,
            sinkhorn_eps=mhc_sinkhorn_eps,
            post_mult_value=mhc_post_mult_value,
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.next_layer_layernorm: Optional[RMSNorm] = None

    def _run_sub_block(
        self,
        residual: torch.Tensor,
        hc: mHC,
        norm: RMSNorm,
        sub_block,
        sub_block_kwargs: Dict,
        apply_fp16_scaling: bool = False,
    ) -> torch.Tensor:
        """Run one mHC sub-block (attn or ffn)."""
        post_mix, comb_mix, layer_input = hc.pre_mapping(residual)
        if self.mhc_h_res_clamp_min is not None:
            comb_mix = comb_mix.clamp(
                min=self.mhc_h_res_clamp_min,
                max=self.mhc_h_res_clamp_max,
            )
        layer_input = norm(layer_input)
        x = sub_block(layer_input, **sub_block_kwargs)
        if apply_fp16_scaling and x.dtype == torch.float16:
            x = x * (1.0 / self.routed_scaling_factor)
        hidden_states = hc.post_mapping(x, residual, post_mix, comb_mix.transpose(-1, -2))
        return hidden_states

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the attention and MLP sub-blocks."""

        def _attn(x_in):
            return self.self_attn(
                position_ids=position_ids,
                hidden_states=x_in,
                attn_metadata=attn_metadata,
                all_reduce_params=AllReduceParams(enable_allreduce=not self.disable_attn_allreduce),
                **kwargs,
            )

        hidden_states = self._run_sub_block(
            residual=hidden_states,
            hc=self.attn_hc,
            norm=self.input_layernorm,
            sub_block=_attn,
            sub_block_kwargs={},
            apply_fp16_scaling=True,
        )

        if self.is_moe:

            def _mlp(x_in):
                return self.mlp(
                    x_in,
                    all_rank_num_tokens=getattr(attn_metadata, "all_rank_num_tokens", None),
                    final_all_reduce_params=AllReduceParams(enable_allreduce=self.tp_size > 1),
                    do_finalize=True,
                )

        else:

            def _mlp(x_in):
                return self.mlp(
                    x_in,
                    final_all_reduce_params=AllReduceParams(enable_allreduce=self.tp_size > 1),
                )

        hidden_states = self._run_sub_block(
            residual=hidden_states,
            hc=self.ffn_hc,
            norm=self.post_attention_layernorm,
            sub_block=_mlp,
            sub_block_kwargs={},
            apply_fp16_scaling=not self.is_moe,
        )

        return hidden_states, None


class XingChen4Model(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        _patch_rope_scaling(config)

        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.n_streams = getattr(config, "hc_mult", getattr(config, "num_residual_streams", 1))

        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList(
            [
                XingChen4DecoderLayer(model_config, layer_idx, self.aux_stream_dict)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the "
                "same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.n_streams > 1:
            hidden_states = inputs_embeds.unsqueeze(1).expand(-1, self.n_streams, -1).contiguous()
        else:
            hidden_states = inputs_embeds

        for layer in self.layers[: self.num_hidden_layers]:
            hidden_states, _ = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
            )

        if self.n_streams > 1:
            hidden_states = hidden_states.mean(dim=1)
        return self.norm(hidden_states)


class XingChen4WeightLoader(DeepseekV3WeightLoader):
    """Load XingChen4 weights and reuse the DeepSeekV3 loader."""

    _HC_STEMS = ("attn_hc", "ffn_hc")

    def _load_hc(self, weights):
        """Load mHC weights from the checkpoint."""
        can_mark_consumed = hasattr(weights, "mark_consumed_keys")

        hc_modules: Dict[str, mHC] = {
            name: module for name, module in self.model.named_modules() if isinstance(module, mHC)
        }

        for name, module in hc_modules.items():
            fn_key = f"{name}.hc_fn"
            base_key = f"{name}.hc_base"
            scale_key = f"{name}.hc_scale"
            if fn_key in weights and base_key in weights and scale_key in weights:
                module.fn.data.copy_(weights[fn_key][:].to(torch.float32).contiguous())
                module.base.data.copy_(weights[base_key][:].to(torch.float32).contiguous())
                module.scale.data.copy_(weights[scale_key][:].to(torch.float32).contiguous())
                if can_mark_consumed:
                    weights.mark_consumed_keys((fn_key, base_key, scale_key))
                continue

            mapping_key = f"{name}.mapping_weight"
            bias_key = f"{name}.bias"
            alpha_keys = (
                f"{name}.alpha_pre",
                f"{name}.alpha_post",
                f"{name}.alpha_res",
            )
            if (
                mapping_key not in weights
                or bias_key not in weights
                or any(key not in weights for key in alpha_keys)
            ):
                continue

            module.fn.data.copy_(weights[mapping_key][:].to(torch.float32).contiguous())
            module.base.data.copy_(weights[bias_key][:].to(torch.float32).contiguous())
            module.scale.data.copy_(
                torch.cat([weights[key][:].reshape(1) for key in alpha_keys])
                .to(torch.float32)
                .contiguous()
            )

            if can_mark_consumed:
                weights.mark_consumed_keys((mapping_key, bias_key, *alpha_keys))

    def load_weights(
        self,
        weights: ConsumableWeightsDict,
        skip_modules: List[str] = [],
    ):
        """Load weights and skip the mHC stems."""
        self._load_hc(weights)
        merged_skip = list(skip_modules) + list(self._HC_STEMS)
        super().load_weights(weights, skip_modules=merged_skip)


@register_auto_model("XingChen4ForCausalLM")
class XingChen4ForCausalLM(SpecDecOneEngineForCausalLM[XingChen4Model, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        _patch_rope_scaling(model_config.pretrained_config)
        super().__init__(
            model=XingChen4Model(model_config),
            model_config=model_config,
        )

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

    def load_weights(self, weights):
        loader = XingChen4WeightLoader(self)
        loader.load_weights(weights)

    def setup_aliases(self) -> None:
        for layer in self.model.layers[: self.config.num_hidden_layers]:
            layer.next_layer_layernorm = None
