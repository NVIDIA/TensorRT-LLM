import copy
from typing import Optional

import torch
from torch import nn
from transformers import Qwen3Config

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.qk_norm_attention import QKNormRoPEAttention
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model


class Qwen3Attention(QKNormRoPEAttention):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
        layer_idx: Optional[int] = None,
        fuse_qk_norm_rope: bool = True,
        attn_output_gate: bool = False,
        use_gemma_rms_norm: bool = False,
        disable_deep_gemm: bool = False,
        reduce_output: bool = True,
        mapping_with_cp: Optional[Mapping] = None,
    ):
        config = model_config.pretrained_config
        self.pretrained_config = config
        self.attn_output_gate = attn_output_gate

        if getattr(config, "rope_scaling", None) is not None:
            if "type" in config.rope_scaling:
                pos_type = config.rope_scaling["type"]
            elif "rope_type" in config.rope_scaling:
                pos_type = config.rope_scaling["rope_type"]
            else:
                raise ValueError(
                    "rope_scaling must have type or rope_type field")
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.from_string(pos_type),
                rope=RopeParams.from_config(config),
                mrope_section=config.rope_scaling.get("mrope_section", None),
                mrope_interleaved=config.rope_scaling.get(
                    "mrope_interleaved", False))
            if config.rope_scaling.get("mrope_interleaved", False):
                fuse_qk_norm_rope = False
        else:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=getattr(config, "attention_bias", None),
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            layer_idx=layer_idx,
            rope_fusion=not getattr(config, 'disable_fuse_rope', False),
            dtype=config.torch_dtype,
            dense_bias=getattr(config, "attention_bias", None),
            config=model_config,
            attn_output_gate=self.attn_output_gate,
            use_gemma_rms_norm=use_gemma_rms_norm,
            disable_deep_gemm=disable_deep_gemm,
            reduce_output=reduce_output,
            mapping_with_cp=mapping_with_cp,
        )


class Qwen3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
        layer_idx: int,
        mapping_with_cp: Optional[Mapping] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        # When enable_attention_dp is True, TP reduction is skipped since each DP rank
        # works on different batch elements. However, with CP > 1, attention is split
        # across CP ranks for the SAME batch element, so reduction is still needed
        # within the CP group.
        needs_tp_reduce = not self.enable_attention_dp and self.mapping.tp_size > 1
        needs_cp_reduce = mapping_with_cp is not None and mapping_with_cp.has_cp_helix(
        )
        self.self_attn = Qwen3Attention(
            model_config,
            layer_idx=layer_idx,
            mapping_with_cp=mapping_with_cp,
            reduce_output=needs_tp_reduce or needs_cp_reduce,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
            dtype=config.torch_dtype,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            config=model_config,
        )

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

        # Attention allreduce: needed unless tp_size==1 or attention_dp
        # without CP. With helix CP, attention ranks process the SAME batch
        # element, so all-reduce across CP ranks is still required.
        has_cp = mapping_with_cp is not None and mapping_with_cp.cp_size > 1
        can_skip_attn_for_dp = self.enable_attention_dp and not has_cp
        self.disable_attn_allreduce = (self.mapping.tp_size == 1
                                       or can_skip_attn_for_dp)

        # MLP allreduce: when enable_attention_dp, MLP uses
        # overridden_tp_size=1 (data-parallel), so no all-reduce is needed.
        self.disable_mlp_allreduce = (self.mapping.tp_size == 1
                                      or self.enable_attention_dp)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        mrope_config: Optional[dict] = None,
        deepstack_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_attn_allreduce),
            mrope_config=mrope_config,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_mlp_allreduce),
            cutlass_min_latency_mode=False,
            **kwargs,
        )
        if deepstack_embeds is not None and self.layer_idx in range(
                len(deepstack_embeds)):
            residual = residual + deepstack_embeds[self.layer_idx]

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        return hidden_states, residual


class Qwen3Model(DecoderModel):

    def __init__(self,
                 model_config: ModelConfig[Qwen3Config],
                 mapping_with_cp: Optional[Mapping] = None):
        super().__init__(model_config)
        config = self.model_config

        self.embed_tokens = Embedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                model_config,
                layer_idx,
                mapping_with_cp=mapping_with_cp,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        mrope_config: Optional[dict] = None,
        # args for deepstack
        deepstack_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mrope_config=mrope_config,
                deepstack_embeds=deepstack_embeds,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Qwen3ForCausalLM")
class Qwen3ForCausalLM(SpecDecOneEngineForCausalLM[Qwen3Model, Qwen3Config]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
    ):
        self.mapping_with_cp = None
        # When helix CP is enabled, CP is relevant only for the attention layer.
        # For other layers (e.g., MLP), CP ranks are repurposed to TP. We save
        # the original mapping with CP, repurpose CP to TP for model construction,
        # and restore the original mapping afterward.
        if model_config.mapping.has_cp_helix():
            self.mapping_with_cp = copy.deepcopy(model_config.mapping)
            model_config._frozen = False
            model_config.mapping = model_config.mapping.repurpose_helix_cp_to_tp(
            )
            model_config._frozen = True

        super().__init__(
            Qwen3Model(model_config, mapping_with_cp=self.mapping_with_cp),
            model_config,
        )

        # Restore the original mapping with CP after model construction.
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True
