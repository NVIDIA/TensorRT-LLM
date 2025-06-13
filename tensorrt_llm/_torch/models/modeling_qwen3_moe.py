import os
from typing import Dict, List, Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import Qwen3MoeConfig

from tensorrt_llm._mnnvl_utils import MnnvlMemory

from ..attention_backend import AttentionMetadata
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           allgather)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (BaseMoeRoutingMethod, CutlassFusedMoE, MoE,
                                 RenormalizeMoeRoutingMethod,
                                 RenormalizeNaiveMoeRoutingMethod,
                                 RoutingMethodType, create_moe)
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import disable_fp4_allgather
from .modeling_qwen3 import Qwen3Attention
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, duplicate_kv_weight,
                             filter_weights, register_auto_model)


class Qwen3Gate(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: Optional[torch.dtype] = None,
        apply_routing: bool = False,
        routing_method_type: RoutingMethodType = RoutingMethodType.Renormalize,
        moe_backend: str = "CUTLASS",
    ):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size),
                                               dtype=dtype),
                                   requires_grad=False)
        self.routing_method_type = routing_method_type
        # FIXME: out_dtype=float32 does not work
        # self.out_dtype = torch.float32 if moe_backend == "TRTLLM" else dtype
        self.out_dtype = dtype

        assert not apply_routing, "Qwen3Gate routing is called inside MoE"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = torch.ops.trtllm.cublas_mm(
            hidden_states, self.weight.t(), bias=None, out_dtype=self.out_dtype)
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

    @property
    def routing_method(self) -> BaseMoeRoutingMethod:
        if self.routing_method_type == RoutingMethodType.RenormalizeNaive:
            return RenormalizeNaiveMoeRoutingMethod(top_k=self.top_k)
        elif self.routing_method_type == RoutingMethodType.Renormalize:
            return RenormalizeMoeRoutingMethod(top_k=self.top_k)
        else:
            raise ValueError(
                f"Unsupported routing method: {self.routing_method_type}")


class Qwen3MoE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3MoeConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.mapping = model_config.mapping
        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy)
        self.enable_alltoall = Qwen3MoE.should_enable_alltoall(
            model_config, self.top_k)
        if self.enable_alltoall:
            MnnvlMemory.initialize()

        self.gate = Qwen3Gate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=config.torch_dtype,
            apply_routing=False,
            routing_method_type=RoutingMethodType.Renormalize,
            moe_backend=model_config.moe_backend,
        )

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            aux_stream=aux_stream,
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
        )

    @staticmethod
    def should_enable_alltoall(model_config: ModelConfig, top_k: int) -> bool:
        if not model_config.mapping.enable_attention_dp:
            return False

        if model_config.mapping.tp_size == 1:
            return False

        if not MnnvlMemory.supports_mnnvl():
            return False

        if os.environ.get("TRTLLM_MOE_DISABLE_ALLTOALLV", "0") == "1":
            return False

        if model_config.mapping.moe_ep_size <= top_k:
            return False

        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        use_dp_padding = False
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.enable_attention_dp and self.mapping.tp_size > 1:
            # FP4 all_gather moves this bf16 allgather in to after topk and fp4 quantization
            # to reduce allreduce BW
            if disable_fp4_allgather() and not self.enable_alltoall:
                hidden_states = allgather(hidden_states,
                                          self.mapping,
                                          dim=0,
                                          sizes=all_rank_num_tokens)
            elif not isinstance(self.experts, CutlassFusedMoE) or (
                    not self.experts.has_fp8_qdq and self.experts.has_nvfp4):
                # Use padding when not using the cutlass path or when x_sf in self.experts is not None
                use_dp_padding = True
                max_num_token = max(all_rank_num_tokens)
                hidden_states = torch.nn.functional.pad(
                    hidden_states,
                    (0, 0, 0, max_num_token - hidden_states.shape[0]))

        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding)

        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            final_hidden_states = self.allreduce(
                final_hidden_states, all_reduce_params=all_reduce_params)

        return final_hidden_states.view(orig_shape)


class Qwen3MoEDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__()
        config = model_config.pretrained_config
        self.self_attn = Qwen3Attention(
            model_config,
            layer_idx=layer_idx,
        )
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = Qwen3MoE(model_config, aux_stream, layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy)
        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get(
            "TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        has_tp = self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )

        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)))

        if self.fusion_config.POST_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                ))
        else:
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)
        return hidden_states, residual


class Qwen3MoEModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):
        super().__init__(model_config)
        config = self.model_config
        self.padding_idx = config.pretrained_config.pad_token_id
        self.aux_stream = torch.cuda.Stream()

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embed_tokens = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype)
        else:
            self.embed_tokens = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )
        self.layers = nn.ModuleList([
            Qwen3MoEDecoderLayer(
                model_config,
                layer_idx,
                self.aux_stream,
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
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual)
        return hidden_states


@register_auto_model("Qwen3MoeForCausalLM")
class Qwen3MoeForCausalLM(DecoderModelForCausalLM[Qwen3MoEModel,
                                                  Qwen3MoeConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3MoeConfig],
    ):
        super().__init__(
            Qwen3MoEModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        enable_attention_dp = self.model_config.mapping.enable_attention_dp

        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads)

        params_map = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"]
        }
        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        "lm_head"):
                    continue

                names = name.split(".")
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights(".".join(names[:-1] + [new_name]),
                                            weights)
                        tensors_need_duplication = ["weight", "bias"]
                        if module.quant_config.quant_mode.has_nvfp4():
                            tensors_need_duplication.append("weight_scale")
                        if new_name in ["k_proj", "v_proj"]:
                            fw = {
                                k: (duplicate_kv_weight(
                                    weight=v[:],
                                    head_dim=head_dim,
                                    tensor_parallel_size=tp_size
                                    if not enable_attention_dp else 1)
                                    if k in tensors_need_duplication else v)
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if isinstance(module, MoE):
                        updated_module_weights = {}
                        for weight_name, weight_value in module_weights.items():
                            new_weight_name = (weight_name.replace(
                                "gate_proj",
                                "w1").replace("up_proj",
                                              "w3").replace("down_proj", "w2"))
                            updated_module_weights[
                                new_weight_name] = weight_value
                        del module_weights
                        module.load_weights(weights=[updated_module_weights])
                    elif hasattr(module, "load_weights"):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
