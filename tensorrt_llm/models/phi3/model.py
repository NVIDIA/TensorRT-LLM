from typing import Optional

import numpy as np
from transformers import AutoModelForCausalLM

from ..._utils import pad_vocab_size
from ...functional import PositionEmbeddingType, Tensor
from ...layers import (MLP, Attention, AttentionMaskType, Embedding,
                       ParallelLMHead, RmsNorm)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig, save_checkpoint)
from .convert import convert_hf_config, convert_hf_weights


class Phi3DecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        position_embedding_type = PositionEmbeddingType.rope_gpt_neox

        rope_scaling_short_factors = 1.0
        rope_scaling_long_factors = 1.0
        original_max_position_embeddings = config.max_position_embeddings
        if hasattr(config, "longrope_scaling_short_factors"):
            rope_scaling_short_factors = np.asarray(
                config.longrope_scaling_short_factors).astype(np.float32)
            rope_scaling_long_factors = np.asarray(
                config.longrope_scaling_long_factors).astype(np.float32)
            original_max_position_embeddings = config.original_max_position_embeddings
            position_embedding_type = PositionEmbeddingType.long_rope

        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            position_embedding_type=position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
            rope_scaling_short_factors=rope_scaling_short_factors,
            rope_scaling_long_factors=rope_scaling_long_factors,
            original_max_position_embeddings=original_max_position_embeddings,
        )

        self.mlp = MLP(hidden_size=config.hidden_size,
                       ffn_hidden_size=config.intermediate_size,
                       hidden_act=config.hidden_act,
                       dtype=config.dtype,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       quant_mode=config.quant_mode,
                       bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
    ):

        input_layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            norm_before_bmm1=True,
        )

        if use_cache:
            attention_output, presents = attention_output

        post_attention_input = hidden_states + attention_output
        post_attention_output = self.post_layernorm(post_attention_input)
        feed_forward_hidden_states = self.mlp(post_attention_output, )
        hidden_states = post_attention_input + feed_forward_hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class Phi3Model(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.vocab_embedding = Embedding(num_embeddings=config.vocab_size,
                                         embedding_dim=config.hidden_size,
                                         dtype=config.dtype)

        self.layers = DecoderLayerList(Phi3DecoderLayer, config)
        self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                            eps=config.norm_epsilon,
                            dtype=config.dtype)

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
    ):
        args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size
                ] if prompt_embedding_table is not None else []
        hidden_states = self.vocab_embedding(input_ids, *args)

        hidden_states = self.layers(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class Phi3ForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = Phi3Model(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        lm_head = ParallelLMHead(config.hidden_size,
                                 vocab_size_padded,
                                 bias=False,
                                 dtype=config.dtype,
                                 tp_group=config.mapping.tp_group,
                                 tp_size=config.mapping.tp_size,
                                 gather_output=True)

        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('rotary_base', 10000.0)

    @classmethod
    def convert_hf_checkpoint(cls,
                              hf_model_dir: str,
                              dtype: Optional[str] = "float16",
                              output_dir: Optional[str] = None,
                              **kwargs):
        '''
        Convert Huggingface checkpoint to TRT-LLM checkpoint
        '''
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                                        torch_dtype="auto",
                                                        trust_remote_code=True)
        config = convert_hf_config(hf_model.config, dtype=dtype, **kwargs)
        weights = convert_hf_weights(hf_model, dtype=dtype, **kwargs)

        if output_dir:
            save_checkpoint(output_dir, config=config, weights=weights)

        return {"weights": weights, "config": config}
