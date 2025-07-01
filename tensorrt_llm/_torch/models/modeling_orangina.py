from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn.parameter import Parameter
from tqdm import tqdm

from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (MoE, MoEWeightLoadingMode,
                                 RenormalizeMoeRoutingMethod, create_moe,
                                 create_renormalize_expert_load_balanced_logits)
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import Fp4QuantizedTensor
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             filter_weights, register_auto_model)


@dataclass
class OranginaModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    # added for TRT-LLM
    torch_dtype: torch.dtype = torch.bfloat16
    rms_norm_eps: float = 1e-05
    # TODO: check what the real max_position_embeddings is
    max_position_embeddings: int = 8192
    model_type: str = "mixtral"
    tie_word_embeddings: bool = False


class AttentionBlock(Attention):

    def __init__(
        self,
        config: ModelConfig[OranginaModelConfig],
        layer_idx: int = 0,
    ):
        pretrained_config = config.pretrained_config

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams(
                dim=pretrained_config.head_dim,
                theta=pretrained_config.rope_theta,
                scale_type=RotaryScalingType.yarn,
                scale=pretrained_config.rope_scaling_factor,
                max_positions=pretrained_config.max_position_embeddings,
                original_max_positions=pretrained_config.initial_context_length,
                beta_fast=pretrained_config.rope_ntk_beta,
                beta_slow=pretrained_config.rope_ntk_alpha,
                duplicate_data=False),
            is_neox=False,
        )

        super().__init__(
            hidden_size=pretrained_config.hidden_size,
            num_attention_heads=pretrained_config.num_attention_heads,
            num_key_value_heads=pretrained_config.num_key_value_heads,
            max_position_embeddings=pretrained_config.max_position_embeddings,
            bias=True,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=pretrained_config.torch_dtype,
            dense_bias=True,
            config=config,
            q_scaling=1.0,
            attention_chunk_size=None,
        )

        # Only apply sliding window to every other layer
        self.sliding_window = pretrained_config.sliding_window if layer_idx % 2 == 0 else None

        self.sinks = Parameter(
            torch.empty(pretrained_config.num_attention_heads // self.tp_size,
                        dtype=torch.float32))
        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor | Fp4QuantizedTensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = hidden_states
        hidden_states = self.norm(hidden_states)

        attention_window_size = self.sliding_window
        attn_output = super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            attention_window_size=attention_window_size,
            attention_sinks=self.sinks.data,
            **kwargs)
        return attn_output + x

    def load_weights(self, weights: Dict):
        sinks = weights[0]['sinks'][self.num_heads *
                                    self.tp_rank:self.num_heads *
                                    (self.tp_rank + 1)]
        self.sinks.data = sinks.to(torch.float32).to("cuda")


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig[OranginaModelConfig],
        layer_idx: int,
    ):
        super().__init__()

        self.config = config  # Store config as instance variable
        pretrained_config = config.pretrained_config
        self.num_experts = pretrained_config.num_experts
        self.layer_idx = layer_idx

        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

        self.gate = Linear(
            in_features=pretrained_config.hidden_size,
            out_features=pretrained_config.num_experts,
            bias=True,
            dtype=pretrained_config.torch_dtype,
            use_custom_cublas_mm=
            False,  # TODO: check perf & cublass mm can not support bias.
        )

        self.routing_method = RenormalizeMoeRoutingMethod(
            top_k=pretrained_config.experts_per_token)
        self.swiglu_alpha = torch.tensor([1.702] * self.num_experts,
                                         dtype=torch.float32).cuda().reshape(
                                             self.num_experts, 1)
        self.swiglu_beta = torch.tensor([1.0] * self.num_experts,
                                        dtype=torch.float32).cuda().reshape(
                                            self.num_experts, 1)

        self.experts = create_moe(
            routing_method=self.routing_method,
            num_experts=pretrained_config.num_experts,
            hidden_size=pretrained_config.hidden_size,
            intermediate_size=pretrained_config.intermediate_size,
            dtype=pretrained_config.torch_dtype,
            reduce_results=True,
            model_config=config,
            weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            bias=True,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
        )

    @staticmethod
    def swiglu(x, alpha: float = 1.702):
        """
        This function is not really used in self.forward(), it's kept here for reference.
        It's implemented as part of the MoE kernels.
        """
        # Note we add an extra bias of 1 to the linear layer
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        return out_glu * (x_linear + 1)

    def _create_ideal_expert_load_balanced_logits(
            self, num_tokens: int, num_experts: int,
            device: torch.device) -> torch.Tensor:
        """
        Create ideal logits that produce GPU-aware load balanced expert assignment.
        This method now delegates to the generic utility function in fused_moe.routing.
        """
        pretrained_config = self.config.pretrained_config
        assert self.config.mapping.moe_tp_size == 1, "this load balance scheme is tested with only MOE EP"

        return create_renormalize_expert_load_balanced_logits(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=pretrained_config.experts_per_token,
            moe_ep_size=self.config.mapping.moe_ep_size,
            device=device,
            dtype=pretrained_config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        # Use ideal load balanced logits if enabled, otherwise use gate output
        if self.config.enable_perfect_router:
            # WARNING: This discards the learned gate output and uses ideal logits for perfect load balancing
            # Only use this for testing load balancing strategies, not for actual inference
            # The gate is still computed to maintain realistic performance measurement
            num_tokens, num_experts = g.shape
            g = self._create_ideal_expert_load_balanced_logits(
                num_tokens=num_tokens, num_experts=num_experts, device=x.device)

        t = self.experts(x=t, router_logits=g)
        return x + t


class TransformerBlock(DecoderLayer):

    def __init__(
        self,
        config: ModelConfig[OranginaModelConfig],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx)
        self.mlp = MLPBlock(config, layer_idx)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.attn(position_ids,
                      hidden_states,
                      attn_metadata,
                      residual=residual,
                      **kwargs)
        x = self.mlp(x)
        return x


class Transformer(DecoderModel):

    def __init__(self, model_config: ModelConfig[OranginaModelConfig]):
        super().__init__(model_config)
        config = self.model_config

        self.embedding = Embedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.block = nn.ModuleList([
            TransformerBlock(
                model_config,
                layer_idx,
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
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mrope_config: Optional[Tuple[torch.Tensor, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        hidden_states = inputs_embeds or self.embedding(input_ids)

        residual = None
        for block in self.block:
            # TODO: apply rms_norm/residual_add fusion
            # hidden_states, residual = block(
            #     position_ids=position_ids,
            #     hidden_states=hidden_states,
            #     attn_metadata=attn_metadata,
            #     residual=residual,
            #     mrope_config=mrope_config,
            # )
            hidden_states = block(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                mrope_config=mrope_config,
            )
            residual = None

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# TODO: fix plumbing for OranginaModelConfig
@register_auto_model("OranginaForCausalLM")
class OranginaForCausalLM(DecoderModelForCausalLM[Transformer,
                                                  OranginaModelConfig]):

    def __init__(
        self,
        model_config: ModelConfig[OranginaModelConfig],
    ):
        super().__init__(
            Transformer(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
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

    def load_weights(self, weights: Dict):
        # TODO: remove the upcast
        for k, v in weights.items():
            if v.dtype == torch.float8_e5m2:
                weights[k] = v.to(self.model.dtype)

        params_map = {
            # TRTLLM module name : Orangina module name
            "qkv_proj": "qkv",
            "o_proj": "out",
            "lm_head": "unembedding",
        }
        head_dim = self.config.head_dim
        num_q_head = self.config.num_attention_heads
        num_kv_head = self.config.num_key_value_heads
        num_expert = self.config.num_experts
        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0:
                continue
            names = name.split(".")
            module_weights = {}
            if names[-1] in params_map:
                names[-1] = params_map[names[-1]]
            # Drop the first "model" prefix
            if names[0] == 'model':
                name = '.'.join(names[1:])
            else:
                name = '.'.join(names)
            module_weights = filter_weights(name, weights)
            if isinstance(module, MoE):
                # [num_experts, intermediate_size * 2, hidden_size]
                gate_up_proj = filter_weights(name.replace("experts", "mlp1"),
                                              weights)
                # [num_experts, intermediate_size, hidden_size]
                down_proj = filter_weights(name.replace("experts", "mlp2"),
                                           weights)
                moe_weights = {
                    'gate_up_proj': [
                        gate_up_proj['weight'][i, :, :].transpose(0, 1)
                        for i in range(num_expert)
                    ],
                    'down_proj': [
                        down_proj['weight'][i, :, :].transpose(0, 1)
                        for i in range(num_expert)
                    ],
                    'gate_up_proj.bias':
                    [gate_up_proj['bias'][i, :] for i in range(num_expert)],
                    'down_proj.bias':
                    [down_proj['bias'][i, :] for i in range(num_expert)]
                }
                module.load_weights(weights=[moe_weights])
            elif hasattr(module, "load_weights"):
                # Load Attention module weights.
                if 'qkv' in name:
                    q_weight = module_weights['weight'][:head_dim *
                                                        num_q_head, :]
                    k_weight = module_weights['weight'][head_dim *
                                                        num_q_head:head_dim *
                                                        (num_q_head +
                                                         num_kv_head), :]
                    v_weight = module_weights['weight'][-head_dim *
                                                        num_kv_head:, :]
                    q_bias = module_weights['bias'][:head_dim * num_q_head]
                    k_bias = module_weights['bias'][head_dim *
                                                    num_q_head:head_dim *
                                                    (num_q_head + num_kv_head)]
                    v_bias = module_weights['bias'][-head_dim * num_kv_head:]
                    qkv_weights = [{
                        'weight': q_weight,
                        'bias': q_bias
                    }, {
                        'weight': k_weight,
                        'bias': k_bias
                    }, {
                        'weight': v_weight,
                        'bias': v_bias
                    }]
                    module.load_weights(weights=qkv_weights)
                else:
                    # Dense & gate & sinks
                    module.load_weights(weights=[module_weights])
            else:
                # Load LN weights.
                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n.replace(
                            "weight", "scale")][:])
