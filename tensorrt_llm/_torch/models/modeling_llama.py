import copy
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import Llama4Config, LlamaConfig

from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, DeepseekAllReduce)
from tensorrt_llm._torch.pipeline_interface import PipelineInterface
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (FusedMoE, Llama4RenormalizeMoeRoutingMethod,
                                 MoEWeightLoadingMode)
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from ..speculative import Eagle3SpecMetadata, SpecMetadata
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, MissingLayer,
                             register_auto_model, support_pp,
                             unpack_hidden_states)


class Llama4Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
        use_qk_norm: bool = False,
        nope_layer: bool = False,
        attn_temperature_tuning: bool = True,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.use_rope = not nope_layer
        self.use_qk_norm = use_qk_norm and not nope_layer
        if self.use_rope and not self.use_qk_norm:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=RopeParams.from_config(config),
                is_neox=False,
            )
        else:
            pos_embd_params = None

        super().__init__(hidden_size=config.hidden_size,
                         num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         bias=config.attention_bias,
                         pos_embd_params=pos_embd_params,
                         layer_idx=layer_idx,
                         dtype=config.torch_dtype,
                         config=model_config)

        if self.use_rope and self.use_qk_norm:
            # here we must disable rope fusion regardless of attn_backend
            self.enable_rope_fusion = False
            self.rotary_emb = RotaryEmbedding(
                RopeParams.from_config(config),
                head_dim=self.head_dim,
                is_neox=False,
            )

        if self.use_qk_norm:
            self.head_dim = config.hidden_size // config.num_attention_heads
            self.qk_norm = RMSNorm(hidden_size=self.head_dim,
                                   eps=1e-6,
                                   dtype=config.torch_dtype,
                                   has_weights=False)
        else:
            self.qk_norm = None

        self.attn_temperature_tuning = attn_temperature_tuning and nope_layer
        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)

    def _attn_qkv(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_metadata: AttentionMetadata,
            attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
            mrope_config: Optional[dict] = None,
            all_reduce_params: Optional[AllReduceParams] = None):
        out_scale = None
        if self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4 or self.o_proj.has_fp8_block_scales:
            out_scale = self.o_proj.inv_input_scale

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.attn.forward(q,
                                        k,
                                        v,
                                        attn_metadata,
                                        out_scale=out_scale,
                                        attention_mask=attention_mask,
                                        mrope_config=mrope_config)

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)

        return attn_output

    def _qk_norm(self, q, k):
        # TODO: make this more efficient.
        q_l2norm = lambda: self.qk_norm(q.reshape(-1, self.head_dim)).reshape(
            -1, self.q_size)
        k_l2norm = lambda: self.qk_norm(k.reshape(-1, self.head_dim)).reshape(
            -1, self.kv_size)
        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def _attention_scaling(self, q, position_ids):

        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def _forward_rope(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        if self.use_qk_norm:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
            assert self.rotary_emb is not None and not self.enable_rope_fusion, "qk_norm requires attention rope fusion disabled"
            q, k = self.rotary_emb(position_ids, [q, k])
            q, k = self._qk_norm(q, k)
            return self._attn_qkv(q, k, v, attn_metadata, attention_mask,
                                  mrope_config, all_reduce_params)
        else:
            # When qk_norm is disabled, use the classic attention path that handles RoPE fusion
            return super().forward(position_ids, hidden_states, attn_metadata,
                                   attention_mask, mrope_config,
                                   all_reduce_params)

    def _forward_nope(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.attn_temperature_tuning:
            q = self._attention_scaling(q, position_ids)
        return self._attn_qkv(q, k, v, attn_metadata, attention_mask,
                              mrope_config, all_reduce_params)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert lora_params is None, "LORA is not supported for Llama4Attention"
        if self.use_rope:
            return self._forward_rope(position_ids, hidden_states,
                                      attn_metadata, attention_mask,
                                      mrope_config, all_reduce_params)
        else:
            return self._forward_nope(position_ids, hidden_states,
                                      attn_metadata, attention_mask,
                                      mrope_config, all_reduce_params)


class LlamaAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class Llama4MoE(nn.Module):

    def __init__(
            self,
            *,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            shared_expert_intermediate_size: int,
            aux_stream: torch.cuda.Stream,
            dtype: Optional[torch.dtype] = None,
            tune_max_num_tokens: int = 8192,
            model_config: ModelConfig = ModelConfig(),
    ):
        from tensorrt_llm._torch.distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.top_k = top_k
        self.experts = FusedMoE(
            routing_method=Llama4RenormalizeMoeRoutingMethod(top_k),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=
            False,  # In both low latency and max-throughput scenarios, FusedMoE needs not to do allreduce inside op.
            weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            model_config=model_config,
            apply_router_weight_on_input=True)

        self.shared_expert = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            reduce_output=False)

        self.router = Linear(hidden_size,
                             num_experts,
                             bias=False,
                             dtype=config.torch_dtype,
                             quant_config=None)

        self.mapping = model_config.mapping
        self.all_reduce = AllReduce(self.mapping)
        self.moe_event = [torch.cuda.Event(), torch.cuda.Event()]
        self.aux_stream = aux_stream

    def compute_routed_output(self, hidden_states, all_rank_num_tokens,
                              min_latency_mode):
        router_logits = self.router(hidden_states)
        routed_output = self.experts(hidden_states, router_logits,
                                     min_latency_mode)
        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        min_latency_mode: Optional[bool] = False,
    ) -> torch.Tensor:
        # Only enable multi-stream for cuda graph since switch stream has extra host overhead
        # This design is mainly for low latency use case. Need to improve for max throughput use case.
        do_multi_stream = torch.cuda.is_current_stream_capturing()
        if do_multi_stream:
            self.moe_event[0].record()
        shared_output = self.shared_expert(hidden_states)
        if do_multi_stream:
            with torch.cuda.stream(self.aux_stream):
                self.moe_event[0].wait()
                routed_output = self.compute_routed_output(
                    hidden_states, all_rank_num_tokens, min_latency_mode)
                self.moe_event[1].record()
            self.moe_event[1].wait()
        else:
            routed_output = self.compute_routed_output(hidden_states,
                                                       all_rank_num_tokens,
                                                       min_latency_mode)
        if min_latency_mode:
            return [shared_output, *routed_output]

        assert shared_output.size() == routed_output.size(
        ), f'unmatched tensor shape'
        final_hidden_states = shared_output + routed_output
        if self.mapping.tp_size > 1:
            final_hidden_states = self.all_reduce(
                final_hidden_states, all_reduce_params=final_all_reduce_params)

        return final_hidden_states


class Llama4DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.is_quanted = model_config.quant_config and model_config.quant_config.quant_mode.has_any_quant(
        )
        self.fusion_config = EagerFusionConfig()
        # self.fusion_config.PRE_MOE_FUSION = model_config.mapping.has_tp(
        # )
        # TODO: re-enable these fusions
        self.fusion_config.PRE_MOE_FUSION = False
        self.fusion_config.POST_MLP_FUSION = False

        self.self_attn = Llama4Attention(
            model_config,
            layer_idx=layer_idx,
            use_qk_norm=getattr(config, "use_qk_norm", False),
            nope_layer=config.no_rope_layers[layer_idx] == 0,
            attn_temperature_tuning=config.attn_temperature_tuning > 0,
            aux_stream=aux_stream,
        )

        is_mlp_layer = (layer_idx + 1) % config.interleave_moe_layer_step != 0

        if is_mlp_layer:
            self.feed_forward = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                # Llama4 has no mlp_bias field.
                bias=getattr(config, "mlp_bias", False),
                dtype=config.torch_dtype,
                config=model_config,
            )

            # self.fusion_config.POST_MLP_FUSION = model_config.mapping.has_tp(
            # )
        else:
            self.feed_forward = Llama4MoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                shared_expert_intermediate_size=config.intermediate_size,
                model_config=model_config,
                aux_stream=aux_stream,
                dtype=config.torch_dtype)

            # self.fusion_config.POST_MOE_FUSION = model_config.mapping.has_tp(
            # )

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

        self.mapping = model_config.mapping
        self.all_reduce = AllReduce(self.mapping)
        self.next_layer_layernorm: RMSNorm = None

        self.deepseek_allreduce = DeepseekAllReduce(self.mapping)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:

        # Only enable min-latency mode on Blackwell
        # TODO: Remove it after we fix crash on Hopper
        # major, minor = torch.cuda.get_device_capability()
        # is_blackwell = (major * 10 + minor) >= 100
        # min_latency_mode = hidden_states.size(
        #     0
        # ) <= 128 and self.fusion_config.POST_MOE_FUSION and is_blackwell and self.is_quanted

        # Temporarily disable min-latency mode for Llama4
        min_latency_mode = False

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.PRE_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            **kwargs,
        )

        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        else:
            # Fully Connected
            hidden_states, residual = unpack_hidden_states(
                self.post_attention_layernorm(hidden_states, residual))

        hidden_states = self.feed_forward(
            hidden_states,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.POST_MOE_FUSION or self.fusion_config.
                POST_MLP_FUSION or self.mapping.tp_size == 1)),
            min_latency_mode=min_latency_mode,
        )
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        if self.fusion_config.POST_MOE_FUSION or self.fusion_config.POST_MLP_FUSION:
            if min_latency_mode:
                shared_output = hidden_states[0]
                hidden_states_activated_experts = hidden_states[1]
                num_activated_experts_per_node = hidden_states[2]
                experts_to_token_score = hidden_states[3]
                activated_expert_global_ids = hidden_states[4]
                hidden_states, residual = self.deepseek_allreduce(
                    hidden_states_activated_experts,  # not used
                    [
                        residual, self.next_layer_layernorm.weight,
                        num_activated_experts_per_node, experts_to_token_score,
                        hidden_states_activated_experts, shared_output,
                        activated_expert_global_ids
                    ],
                    self.next_layer_layernorm.variance_epsilon,
                    AllReduceFusionOp.MOE_ALLREDUCE_RESIDUAL_RMS_NORM,
                )
            else:
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
        elif self.next_layer_layernorm:
            hidden_states, residual = unpack_hidden_states(
                self.next_layer_layernorm(hidden_states, residual))

        return hidden_states, residual


class LlamaDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = LlamaAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
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
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)
        return hidden_states, residual


class Eagle3LlamaAttention(LlamaAttention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
    ):
        super().__init__(model_config, layer_idx)

        model_config = model_config or ModelConfig()
        config = model_config.pretrained_config

        tp_size = model_config.mapping.tp_size

        # Override the QKV projection. The number of input features
        # is twice as big for EAGLE3 draft models.
        self.qkv_proj = Linear(
            2 * self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=config.attention_bias,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=model_config.get_quant_config(),
            skip_create_weights=model_config.skip_create_weights,
        )


class Eagle3LlamaDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = Eagle3LlamaAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.hidden_norm = RMSNorm(hidden_size=config.hidden_size,
                                   eps=config.rms_norm_eps,
                                   dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states

        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_pp
class Llama4Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Llama4DecoderLayer(
                model_config,
                layer_idx,
                self.aux_stream,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> torch.Tensor:
        if self.model_config.mapping.is_first_pp_rank():
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds
            residual = None
        else:
            if pipeline_interface is None:
                raise ValueError(
                    "pipeline_interface is required for non-first pp rank.")
            hidden_states, residual = pipeline_interface
            hidden_states, residual = self.local_layers()[0].input_layernorm(
                hidden_states, residual)

        for decoder_layer in self.local_layers():
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        if self.model_config.mapping.is_last_pp_rank():
            return hidden_states
        else:
            return PipelineInterface(hidden_states, residual)


@support_pp
class LlamaModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> torch.Tensor:
        if self.model_config.mapping.is_first_pp_rank():
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds
            residual = None
        else:
            if pipeline_interface is None:
                raise ValueError(
                    "pipeline_interface is required for non-first pp rank.")
            hidden_states, residual = pipeline_interface

        for decoder_layer in self.local_layers():
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        if self.model_config.mapping.is_last_pp_rank():
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return PipelineInterface(hidden_states, residual)


@register_auto_model("LlamaForCausalLM")
class LlamaForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        super().__init__(LlamaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)


@register_auto_model("Llama4ForConditionalGeneration")
class Llama4ForConditionalGeneration(DecoderModelForCausalLM[Llama4Model,
                                                             Llama4Config]):

    def __init__(
        self,
        model_config: ModelConfig[Llama4Config],
    ):
        # TODO: figure out a better way to handle multimodality.
        model_config = copy.copy(model_config)
        architectures = model_config.pretrained_config.architectures
        model_config.pretrained_config = model_config.pretrained_config.text_config
        model_config.pretrained_config.architectures = architectures
        super().__init__(Llama4Model(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict):
        new_weights = {}
        for key, tensor in weights.items():
            if key.startswith("language_model."):
                new_key = key[len("language_model."):]
                new_weights[new_key] = tensor
            else:
                new_weights[key] = tensor

        super().load_weights(new_weights)

        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            elif not isinstance(self.model.layers[idx + 1], MissingLayer):
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm


@register_auto_model("MistralForCausalLM")
class MistralForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        # to support MistralConfig
        if not hasattr(model_config.pretrained_config, 'attention_bias'):
            model_config.pretrained_config.attention_bias = False
        if not hasattr(model_config.pretrained_config, 'rope_scaling'):
            model_config.pretrained_config.rope_scaling = None
        if not hasattr(model_config.pretrained_config, 'mlp_bias'):
            model_config.pretrained_config.mlp_bias = False

        super().__init__(LlamaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)


class Eagle3LlamaDraftModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]) -> None:
        super().__init__(model_config)

        config = model_config.pretrained_config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = Linear(self.hidden_size_in * 3,
                         config.hidden_size,
                         bias=False,
                         dtype=config.torch_dtype)

        self.midlayer = Eagle3LlamaDecoderLayer(model_config, 0)

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

        self.d2t = nn.Parameter(torch.empty((config.draft_vocab_size, ),
                                            dtype=torch.int64),
                                requires_grad=False)

        if self.hidden_size_in != config.hidden_size:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                parallel_config=ParallelConfig(
                    tensor_parallel_rank=model_config.mapping.tp_rank,
                    tensor_parallel_size=model_config.mapping.tp_size,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    pipeline_parallel_size=model_config.mapping.pp_size,
                    parallel_rank=model_config.mapping.rank,
                    gather_output=True,
                    gpus_per_node=model_config.mapping.gpus_per_node,
                ),
            )
        else:
            # Shared with target model.
            self.embed_tokens = None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.embed_tokens is not None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).to(self.dtype)

        assert hidden_states is not None

        # NOTE: If hidden states from the target model have to be concatenated,
        # we expect that to happen outside the model definition. This helps us
        # avoid data-dependent control flow and gives us better CUDA graph
        # coverage.
        hidden_states, residual = self.midlayer(position_ids=position_ids,
                                                embeds=inputs_embeds,
                                                hidden_states=hidden_states,
                                                attn_metadata=attn_metadata)

        hidden_states, hidden_states_to_save = self.norm(
            hidden_states, residual)
        assert isinstance(spec_metadata, Eagle3SpecMetadata)
        spec_metadata.maybe_capture_hidden_states(1, hidden_states_to_save)
        return hidden_states


@register_auto_model("EAGLE3LlamaForCausalLM")
class Eagle3LlamaForCausalLM(DecoderModelForCausalLM[Eagle3LlamaDraftModel,
                                                     LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        super().__init__(
            Eagle3LlamaDraftModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.draft_vocab_size)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            hidden_states=hidden_states,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict):
        new_weights = {}
        for k, v in weights.items():
            new_k = "model." + k if 'lm_head' not in k else k
            new_weights[new_k] = v

        super().load_weights(new_weights)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens

    # TODO: should input/position IDs be included in this? Keeping it implicit
    # for now since the shapes/dtypes are the same across all models we have.
    def get_warmup_extra_inputs(self, batch_size: int,
                                num_tokens: int) -> Dict[str, Any]:

        hidden_states = torch.empty(batch_size * num_tokens,
                                    self.model.hidden_size_in,
                                    dtype=self.model.dtype,
                                    device='cuda')

        return {'hidden_states': hidden_states}

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Hack for eagle3. We might need to run a matmul to reduce
        the dimensionality of the hidden states on the first pass
        through the draft model. Shape dependent control flow will
        not work with CUDA graphs. So we have hoisted this logic out
        of the forward pass - the pyexecutor will call this function
        before running forward when applicable.
        """
        hidden_states = hidden_states.to(self.model.dtype)

        expected_hidden_size = self.model.hidden_size
        if hidden_states.shape[-1] != expected_hidden_size:
            hidden_states = self.model.fc(hidden_states)

        return hidden_states
