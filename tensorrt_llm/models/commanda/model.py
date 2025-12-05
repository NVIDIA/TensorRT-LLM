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
from .config import Cohere2Config


class Cohere2DecoderLayer(Module):

    def __init__(self, config: Cohere2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Every n-th layer uses global attention with no position embeddings.
        # The other layers use sliding window attention with rotary embeddings.
        self.sliding_window = (self.layer_idx + 1) % 4 != 0;

        self.input_layernorm = LayerNorm(
            normalized_shape=config.hidden_size,
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
            attention_mask_type=AttentionMaskType.sliding_window_causal if self.sliding_window else AttentionMaskType.causal,
            bias=config.attn_bias,
            # There isn't a "none" option for the position embedding type.
            # So this is theoretically incorrect, there are no learned absolute position embeddings.
            # But because we only set it on the layer, rather than the entire model, it ends up applying no position embeddings.
            position_embedding_type=PositionEmbeddingType.rope_gptj if self.sliding_window else PositionEmbeddingType.learned_absolute,
            rotary_embedding_base=config.rotary_base,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
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


class Cohere2Model(Module):

    def __init__(self, config: Cohere2Config) -> None:
        super().__init__()

        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.dtype,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                tp_rank=config.mapping.tp_rank)

        self.layers = DecoderLayerList(Cohere2DecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(
                normalized_shape=config.hidden_size,
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
            hidden_states=None):

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


class Cohere2ForCausalLM(DecoderModelForCausalLM):

    config_class = Cohere2Config

    def __init__(self, config: Cohere2Config):
        transformer = Cohere2Model(config)
        vocab_size_padded = pad_vocab_size(
            config.vocab_size,
            config.mapping.tp_size)

        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(
                config.hidden_size,
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
    def from_hugging_face(
            cls,
            hf_model_or_dir: str,
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):

        config = Cohere2Config.from_hugging_face(
            hf_model_or_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs)

        model = cls(config)
        loader = ModelWeightsLoader(hf_model_or_dir)
        loader.generate_tllm_weights(model)

        return model
