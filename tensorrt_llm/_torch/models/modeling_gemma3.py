import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import Gemma3TextConfig
from transformers.activations import ACT2FN

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             duplicate_kv_weight, filter_weights,
                             register_auto_model)


class Gemma3TextScaledWordEmbedding(Embedding):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
    ):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
        )
        self.embed_scale = torch.sqrt(torch.tensor(hidden_size)).to(self.dtype)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Gemma3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: Optional[int] = None,
        is_sliding: bool = False,
    ):
        self.is_sliding = is_sliding
        config = model_config.pretrained_config
        rope_params = RopeParams.from_config(config)
        self.attention_window_size = None
        if is_sliding:
            rope_params.theta = 10000
            self.attention_window_size = config.sliding_window
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )
        q_scaling = math.sqrt(config.query_pre_attn_scalar) / math.sqrt(
            config.head_dim)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            q_scaling=q_scaling,
        )
        self.q_norm = RMSNorm(hidden_size=config.head_dim,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)
        self.k_norm = RMSNorm(hidden_size=config.head_dim,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)
        self.aux_stream = torch.cuda.Stream()
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        attention_window_size = self.attention_window_size or attn_metadata.max_seq_len
        return super().forward(position_ids=position_ids,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata,
                               attention_mask=attention_mask,
                               mrope_config=mrope_config,
                               all_reduce_params=all_reduce_params,
                               lora_params=lora_params,
                               attention_window_size=attention_window_size,
                               **kwargs)

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.q_norm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.k_norm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        # Gemma3 applies QK norm before RoPE.
        q, k, v = self.split_qkv(q, k, v)
        q, k = self.apply_qk_norm(q, k)
        return super().apply_rope(q, k, v, position_ids)


class Gemma3MLP(nn.Module):

    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dtype = config.torch_dtype
        self.gate_proj = Linear(self.hidden_size,
                                self.intermediate_size,
                                bias=False,
                                dtype=self.dtype)
        self.up_proj = Linear(self.hidden_size,
                              self.intermediate_size,
                              bias=False,
                              dtype=self.dtype)
        self.down_proj = Linear(self.intermediate_size,
                                self.hidden_size,
                                bias=False,
                                dtype=self.dtype)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
        self.self_attn = Gemma3Attention(
            model_config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
        )

        self.mlp = Gemma3MLP(config)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                 eps=config.rms_norm_eps,
                                                 dtype=config.torch_dtype)
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Gemma3TextConfig]):
        super().__init__(model_config)
        config = self.model_config
        self.hidden_size = config.pretrained_config.hidden_size
        self.padding_idx = config.pretrained_config.pad_token_id

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.pretrained_config.hidden_size,
                            eps=config.pretrained_config.rms_norm_eps,
                            dtype=config.pretrained_config.torch_dtype)

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

        hidden_states = inputs_embeds.to(self.dtype)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(position_ids=position_ids,
                                          hidden_states=hidden_states,
                                          attn_metadata=attn_metadata)

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("Gemma3ForCausalLM")
class Gemma3ForCausalLM(DecoderModelForCausalLM[Gemma3TextModel,
                                                Gemma3TextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
    ):
        super().__init__(Gemma3TextModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
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

    # This is a modified version of the load_weights function in modeling_utils.py with the
    # minor change for Gemma3 RMSNorm.
    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads)

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

                # Skip loading weights for embedding and lm_head if LoRA is enabled.
                if hasattr(
                        self.model_config, 'lora_config'
                ) and self.model_config.lora_config is not None and len(
                        self.model_config.lora_config.lora_dir) == 1 and (
                            name == "model.embed_tokens" or name == "lm_head"):
                    continue

                names = name.split('.')
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    head_dim=head_dim,
                                    tensor_parallel_size=tp_size)
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }

                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                # Gemma3 RMSNorm uses +1 just like LayerNorm-1P.
                                if 'norm' in names[-1]:
                                    p.data.copy_(module_weights[n][:] + 1)
                                else:
                                    p.data.copy_(module_weights[n][:])
