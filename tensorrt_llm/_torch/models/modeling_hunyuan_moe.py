from typing import Dict, Optional, Union

import torch
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (CutlassFusedMoE, RenormalizeMoeRoutingMethod,
                                 VanillaMoE, create_moe)
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             duplicate_kv_weight, register_auto_model)


class HunyuanMoE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size[0] \
            if isinstance(config.moe_intermediate_size, list) else config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk[0] \
            if isinstance(config.moe_topk, list) else config.moe_topk
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        # moe gate (linear layer) only runs in half/full precision for now
        self.gate = Linear(self.hidden_dim,
                           self.num_experts,
                           bias=False,
                           dtype=config.torch_dtype)

        reduce_results = True

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=RenormalizeMoeRoutingMethod(top_k=self.top_k),
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            dtype=config.torch_dtype,
            reduce_results=reduce_results,
            model_config=model_config)

        self.shared_mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, 'mlp_bias') else False,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        shared_expert_output = self.shared_mlp(hidden_states)
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False)

        final_hidden_states = shared_expert_output + final_hidden_states

        return final_hidden_states.view(orig_shape)


class HunYuanAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        use_qk_norm: bool = True,
        nope_layer: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config

        self.use_rope = not nope_layer
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(config),
            is_neox=True,
        ) if self.use_rope else None
        self.use_qk_norm = use_qk_norm

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            rope_fusion=not self.use_qk_norm,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.query_layernorm = RMSNorm(hidden_size=self.head_dim,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.key_layernorm = RMSNorm(hidden_size=self.head_dim,
                                     eps=config.rms_norm_eps,
                                     dtype=config.torch_dtype)
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        q, k, v = self.split_qkv(q, k, v)
        if position_ids is not None:
            q, k, v = super().apply_rope(q, k, v, position_ids)
        # Llama4 applies QK norm after RoPE.
        if self.use_qk_norm:
            q, k = self.apply_qk_norm(q, k)

        return q, k, v

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.query_layernorm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.key_layernorm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert lora_params is None, "LORA is not supported for HunYuanAttention"
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            **kwargs,
        )


class HunYuanDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        # attention
        self.self_attn = HunYuanAttention(
            model_config,
            layer_idx=layer_idx,
        )

        is_experts_valid = ((isinstance(config.num_experts, int)
                             and config.num_experts > 1)
                            or (isinstance(config.num_experts, list)
                                and max(config.num_experts) > 1))
        is_moe_single_node = is_experts_valid and layer_idx >= config.moe_layer_num_skipped  # only support one node yet

        if is_moe_single_node:
            self.mlp = HunyuanMoE(
                model_config, aux_stream_dict[AuxStreamType.MoeChunkingOverlap])
        else:
            self.mlp = GatedMLP(hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                bias=config.mlp_bias,
                                dtype=config.torch_dtype,
                                config=model_config)

        norm_type = getattr(config, 'norm_type', 'rms')
        if norm_type == 'hf_rms' or norm_type == 'rms':
            self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                           eps=config.rms_norm_eps,
                                           dtype=config.torch_dtype)
            self.post_attention_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype)
        elif norm_type == 'fused' or norm_type == 'torch_nn':
            self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        # Fully Connected
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, attn_metadata)
        hidden_states = residual + hidden_states
        return hidden_states


class HunYuanModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.aux_stream_dict = {
            key: torch.cuda.Stream()
            for key in [
                AuxStreamType.Attention, AuxStreamType.MoeShared,
                AuxStreamType.MoeChunkingOverlap
            ]
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList([
            HunYuanDecoderLayer(model_config, layer_idx, self.aux_stream_dict)
            for layer_idx in range(config.num_hidden_layers)
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
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer_idx, decoder_layer in enumerate(self.layers):
            kwargs['layer_idx'] = layer_idx
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("HunYuanMoEV1ForCausalLM")
class HunYuanMoEV1ForCausalLM(DecoderModelForCausalLM[HunYuanModel,
                                                      PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(HunYuanModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
        self._execution_stats = None
        print("---debug model_config: ", model_config)

    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        }
        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        "lm_head"):
                    continue
                names = name.split('.')
                if names[-1] in params_map:
                    # model.layers.{idx}.mlp.shared_mlp.gate_up_proj or model.layers.{idx}.self_attn.qkv_proj
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    num_kv_heads=v[:].shape[0] // head_dim,
                                    tensor_parallel_size=tp_size)
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    name = name.replace('gate', 'gate.wg')
                    module_weights = filter_weights(name, weights)
                    if isinstance(module, CutlassFusedMoE) or isinstance(
                            module, VanillaMoE):
                        # model.layers.{idx}.mlp.experts
                        updated_module_weights = {}
                        for weight_name, weight_value in module_weights.items():
                            new_weight_name = weight_name.replace(
                                "gate_proj",
                                "w1").replace("up_proj",
                                              "w3").replace("down_proj", "w2")
                            updated_module_weights[
                                new_weight_name] = weight_value
                        del module_weights
                        module.load_weights(weights=[updated_module_weights])
                    elif hasattr(module, 'load_weights'):
                        # model.layers.{idx}.self_attn.o_proj or model.layers.{idx}.mlp.shared_mlp.down_proj
                        # or model.layers.{idx}.mlp.experts.gate
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
