from typing import Optional

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import recv, send
from ...layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                       GatedMLP, LayerNorm, PositionEmbeddingType)
from ...mapping import Mapping
from ...module import Module
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import CohereConfig


class CohereDecoderLayer(Module):

    def __init__(self, config: CohereConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                         eps=config.norm_epsilon,
                                         bias=False,
                                         dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        self.local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=self.local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gptj,
            rotary_embedding_base=config.rotary_base,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
            qk_layernorm=config.qk_layernorm,
            layernorm_share=False,
            eps=config.norm_epsilon,
            quant_mode=config.quant_mode)

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            ffn_hidden_size=config.intermediate_size,
                            hidden_act=config.hidden_act,
                            dtype=config.dtype,
                            bias=False,
                            tp_group=config.mapping.tp_group,
                            tp_size=config.mapping.tp_size,
                            quant_mode=config.quant_mode)

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None):
        assert not (
            default_net().plugin_config.reduce_fusion
        ), "Custom all reduce and residual mlp can't be enabled at the same time."

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output

        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + attention_output + mlp_output

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class CohereModel(Module):

    def __init__(self, config: CohereConfig) -> None:
        super().__init__()

        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype,
                                             tp_group=config.mapping.tp_group,
                                             tp_size=config.mapping.tp_size,
                                             tp_rank=config.mapping.tp_rank)

        self.layers = DecoderLayerList(CohereDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  eps=config.norm_epsilon,
                                  bias=False,
                                  dtype=config.dtype)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        use_cache=False,
        attention_mask=None,
        spec_decoding_params=None,
        kv_cache_params=None,
        attention_params=None,
        hidden_states=None,
    ):
        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            spec_decoding_params=spec_decoding_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class CohereForCausalLM(DecoderModelForCausalLM):
    config_class = CohereConfig

    def __init__(self, config: CohereConfig):
        transformer = CohereModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(cls,
                          hf_model_or_dir: str,
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          quant_config: Optional[QuantConfig] = None,
                          **kwargs):
        ''' Create a CohereForCausalLM object from give parameters
        '''

        config = CohereConfig.from_hugging_face(hf_model_or_dir,
                                                dtype=dtype,
                                                mapping=mapping,
                                                quant_config=quant_config,
                                                **kwargs)
        model = cls(config)
        custom_dict = {
            'q_layernorm': 'q_norm',
            'k_layernorm': 'k_norm',
        }
        loader = ModelWeightsLoader(hf_model_or_dir, custom_dict)
        loader.generate_tllm_weights(model)

        return model
