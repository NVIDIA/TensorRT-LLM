from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention, QkNormType
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (MoE, MoEWeightLoadingMode,
                                 RenormalizeMoeRoutingMethod, create_moe)
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import Fp4QuantizedTensor
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             filter_weights, register_auto_model)


@dataclass
class OranginaModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    rope_theta: float = 150000.0

    # added for TRT-LLM
    torch_dtype: torch.dtype = torch.bfloat16
    rms_norm_eps: float = 1e-05
    num_experts_per_tok: int = 4
    # TODO: check what the real max_position_embeddings is
    max_position_embeddings: int = 8192
    model_type: str = "orangina"
    tie_word_embeddings: bool = False


class AttentionBlock(Attention):

    def __init__(
        self,
        config: ModelConfig[OranginaModelConfig],
        layer_idx: int = 0,
    ):
        pretrained_config = config.pretrained_config

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(pretrained_config),
        )

        super().__init__(
            hidden_size=pretrained_config.hidden_size,
            num_attention_heads=pretrained_config.num_attention_heads,
            num_key_value_heads=pretrained_config.num_key_value_heads,
            max_position_embeddings=pretrained_config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            qk_norm_type=QkNormType.none,
            layer_idx=layer_idx,
            dtype=pretrained_config.torch_dtype,
            dense_bias=False,
            config=config,
            q_scaling=1.0,
            attention_chunk_size=None,
        )

        # Only apply sliding window to every other layer
        self.sliding_window = pretrained_config.sliding_window if layer_idx % 2 == 0 else None

        # TODO: pass sinks to Attention kernel
        self.sinks = torch.empty(pretrained_config.num_attention_heads,
                                 dtype=torch.bfloat16)
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
        hidden_states = self.norm(hidden_states)

        attention_window_size = self.sliding_window
        return super().forward(position_ids=position_ids,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata,
                               attention_mask=attention_mask,
                               all_reduce_params=all_reduce_params,
                               lora_params=lora_params,
                               attention_window_size=attention_window_size,
                               **kwargs)


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig[OranginaModelConfig],
        layer_idx: int,
    ):
        super().__init__()

        pretrained_config = config.pretrained_config
        self.num_experts = pretrained_config.num_experts
        self.layer_idx = layer_idx

        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

        self.gate = Linear(
            in_features=pretrained_config.hidden_size,
            out_features=pretrained_config.num_experts,
            dtype=pretrained_config.torch_dtype,
            use_custom_cublas_mm=
            False,  # TODO: check perf & cublass mm can not support bias.
        )

        self.routing_method = RenormalizeMoeRoutingMethod(
            top_k=pretrained_config.num_experts_per_tok)

        # TODO: route "block.x.mlp" weights to "block.x.mlp.experts" in weight loading
        self.experts = create_moe(
            routing_method=self.routing_method,
            num_experts=pretrained_config.num_experts,
            hidden_size=pretrained_config.hidden_size,
            intermediate_size=pretrained_config.intermediate_size,
            dtype=pretrained_config.torch_dtype,
            reduce_results=True,
            model_config=config,
            weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)

        # TODO: add bias and customized swiglu to MoE kernels
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
        # TODO: add unembedding, which is implemented as lm_head in DecoderModelForCausalLM

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
            "sinks": "sdpa.sinks",
            # "experts": ["mlp1", "mlp2"]
        }
        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0:
                continue
            names = name.split(".")
            if names[-1] in params_map:
                names[-1] = params_map[names[-1]]
            # Drop the first "model" prefix
            if names[0] == 'model':
                name = '.'.join(names[1:])
            else:
                name = '.'.join(names)
            module_weights = filter_weights(name, weights)
            # TODO: Make sure MoE is fused or not.
            if isinstance(module, MoE):
                pass
            elif isinstance(module, AttentionBlock):
                module_weight = filter_weights(name, weights)
                module.sinks.data = module_weight[name]
            # TODO: Make sure QKV is fused or not.
            elif hasattr(module, "load_weights"):
                # Load Attention module weights.
                if 'qkv' in name:
                    continue
                module.load_weights(weights=[module_weights])
            else:
                # Load LN weights.
                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n.replace(
                            "weight", "scale")][:])
