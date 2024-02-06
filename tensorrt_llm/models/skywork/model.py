# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..._utils import pad_vocab_size
from ...functional import Tensor
from ...layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                       GatedMLP, RmsNorm)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)


class SkyworkDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        hidden_size = config.hidden_size
        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        mlp_hidden_size = config.mlp_hidden_size
        hidden_act = config.hidden_act
        position_embedding_type = config.position_embedding_type
        rotary_base = config.rotary_base
        quant_mode = config.quant_mode

        rotary_scaling = None
        if hasattr(config, "rotary_scaling"):
            rotary_scaling = config.rotary_scaling
        if rotary_scaling and rotary_scaling["type"] == "ntk":
            rotary_base *= rotary_scaling["factor"]
            rotary_scaling = None

        self.input_layernorm = RmsNorm(normalized_shape=hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)
        self.attention = Attention(
            hidden_size,
            config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type=position_embedding_type,
            rotary_embedding_base=rotary_base,
            rotary_embedding_scaling=rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=config.mapping.tp_rank,
            quant_mode=quant_mode,
        )
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=dtype)
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=hidden_act,
            dtype=dtype,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=quant_mode,
        )

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        self.register_network_output("decoder_outputs", hidden_states)
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class SkyworkModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        mapping = config.mapping
        self.vocab_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            tp_size=mapping.tp_size if config.use_parallel_embedding else 1,
            tp_group=mapping.tp_group
            if config.use_parallel_embedding else None,
            tp_rank=mapping.tp_rank,
            sharding_dim=config.embedding_sharding_dim)
        self.layers = DecoderLayerList(SkyworkDecoderLayer, config)
        self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                            eps=config.norm_epsilon,
                            dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None):
        # TODO: Add Prompt Tuning support
        hidden_states = self.vocab_embedding(input_ids)
        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params)
        if use_cache:
            hidden_states, presents = hidden_states
        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return hidden_states, tuple(presents)
        return hidden_states


class SkyworkForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        mapping = config.mapping
        transformer = SkyworkModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size, mapping.tp_size)
        lm_head = ColumnLinear(
            config.hidden_size,
            vocab_size_padded,
            bias=False,
            dtype=config.dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )
        super().__init__(config, transformer, lm_head)

    def check_config(self):
        self.config.set_if_not_exist('rope_theta', 10000)
        self.config.set_if_not_exist('rotary_scaling', None)
