from typing import Optional, Union

import numpy as np
from transformers import AutoModelForCausalLM

from ..._utils import pad_vocab_size
from ...functional import PositionEmbeddingType, Tensor
from ...layers import (MLP, MOE, Attention, AttentionMaskType,
                       BlockSparseAttnParams, ColumnLinear, Embedding,
                       LayerNorm, MoeConfig, RmsNorm)
from ...lora_helper import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig, QuantConfig)
from .config import Phi3Config
from .convert import load_weights_from_hf_model


class Phi3DecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        attention_mask_type = AttentionMaskType.causal
        block_sparse_attn_params = BlockSparseAttnParams()
        q_scaling = 1.0
        self.gegelu_limit = None

        self.small_variant = config.architecture == "Phi3SmallForCausalLM"
        self.moe_variant = config.architecture == "PhiMoEForCausalLM"
        if self.small_variant:
            self.gegelu_limit = config.gegelu_limit

            # MuP uses norm_factor=attention_head_size (rather than sqrt(attention_head_size))
            # We achieve this using q_scaling = sqrt(attention_head_size)
            hidden_size = config.hidden_size
            num_attention_heads = config.num_attention_heads
            attention_head_size = hidden_size / num_attention_heads
            q_scaling = attention_head_size**.5

            block_sparse = ((layer_idx + 1) %
                            config.dense_attention_every_n_layers) != 0
            attention_mask_type = AttentionMaskType.blocksparse if block_sparse else AttentionMaskType.causal

            block_sparse_attn_params = BlockSparseAttnParams(
                config.blocksparse_block_size,
                config.blocksparse_homo_head_pattern,
                config.blocksparse_num_local_blocks,
                config.blocksparse_vertical_stride)

        if self.small_variant or self.moe_variant:
            self.input_layernorm = LayerNorm(
                normalized_shape=config.hidden_size,
                dtype=config.dtype,
                eps=config.norm_epsilon)
            self.post_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                            dtype=config.dtype,
                                            eps=config.norm_epsilon)
        else:
            self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                           eps=config.norm_epsilon,
                                           dtype=config.dtype)
            self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                          eps=config.norm_epsilon,
                                          dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        position_embedding_type = PositionEmbeddingType.rope_gpt_neox

        rope_scaling_short_factors, rope_scaling_long_factors = None, None
        rope_scaling_short_mscale, rope_scaling_long_mscale = None, None
        original_max_position_embeddings = config.max_position_embeddings

        if hasattr(config, "longrope_scaling_short_factors"):
            rope_scaling_short_factors = np.asarray(
                config.longrope_scaling_short_factors).astype(np.float32)
            rope_scaling_long_factors = np.asarray(
                config.longrope_scaling_long_factors).astype(np.float32)

            original_max_position_embeddings = config.original_max_position_embeddings
            position_embedding_type = PositionEmbeddingType.long_rope

            if self.small_variant or self.moe_variant:
                rope_scaling_short_mscale = config.longrope_short_mscale
                rope_scaling_long_mscale = config.longrope_long_mscale

        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embedding_type=position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=attention_mask_type,
            bias=self.small_variant or self.moe_variant,
            q_scaling=q_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
            rope_scaling_short_factors=rope_scaling_short_factors,
            rope_scaling_long_factors=rope_scaling_long_factors,
            rope_scaling_short_mscale=rope_scaling_short_mscale,
            rope_scaling_long_mscale=rope_scaling_long_mscale,
            original_max_position_embeddings=original_max_position_embeddings,
            rotary_embedding_percentage=config.rotary_pct,
            block_sparse_params=block_sparse_attn_params)

        ClsMLP = MLP
        mlp_kwargs = {}
        if hasattr(config, "moe"):
            ClsMLP = MOE
            moe_config = MoeConfig()
            for key, value in config.moe.items():
                setattr(moe_config, key, value)
            mlp_kwargs = {
                "moe_config": moe_config,
                "mapping": config.mapping,
            }

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=config.intermediate_size,
                          hidden_act=config.hidden_act,
                          dtype=config.dtype,
                          tp_group=tp_group,
                          tp_size=tp_size,
                          quant_mode=config.quant_mode,
                          bias=self.small_variant,
                          **mlp_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
    ):

        input_layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            norm_before_bmm1=not self.small_variant,
            lora_layer_params=lora_layer_params,
        )

        if use_cache:
            attention_output, presents = attention_output

        post_attention_input = hidden_states + attention_output
        post_attention_output = self.post_layernorm(post_attention_input)
        if self.small_variant:
            feed_forward_hidden_states = self.mlp(
                post_attention_output,
                gegelu_limit=self.gegelu_limit,
                lora_layer_params=lora_layer_params)
        else:
            feed_forward_hidden_states = self.mlp(
                post_attention_output, lora_layer_params=lora_layer_params)
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
        self.small_variant = config.architecture == "Phi3SmallForCausalLM"
        self.moe_variant = config.architecture == "PhiMoEForCausalLM"
        if self.small_variant or self.moe_variant:
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  eps=config.norm_epsilon,
                                  dtype=config.dtype)
            if self.small_variant:
                self.mup_embedding_multiplier = config.mup_embedding_multiplier
        else:
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
        lora_params=None,
    ):
        args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size
                ] if prompt_embedding_table is not None else []
        hidden_states = self.vocab_embedding(input_ids, *args)

        if self.small_variant and self.mup_embedding_multiplier > 0.0:
            hidden_states = hidden_states * self.mup_embedding_multiplier

        hidden_states = self.layers(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
        )
        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class Phi3ForCausalLM(DecoderModelForCausalLM):
    config_class = Phi3Config

    def __init__(self, config: PretrainedConfig):
        transformer = Phi3Model(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        self.moe_variant = config.architecture == "PhiMoEForCausalLM"
        lm_head = ColumnLinear(config.hidden_size,
                               vocab_size_padded,
                               bias=self.moe_variant,
                               dtype=config.dtype,
                               tp_group=config.mapping.tp_group,
                               tp_size=config.mapping.tp_size,
                               gather_output=True)

        if self.moe_variant:
            self.trtllm_modules_to_hf_modules = {
                "attn_q": "q_proj",
                "attn_k": "k_proj",
                "attn_v": "v_proj",
                "attn_dense": "o_proj",
                "moe_h_to_4h": "w1",
                "moe_4h_to_h": "w2",
                "moe_gate": "w3",
                "moe_router": "gate",
            }
        else:
            self.trtllm_modules_to_hf_modules = {
                "attn_qkv": ["qkv_proj", "query_key_value"],
                "attn_dense": ["o_proj", "dense"],
                "mlp_h_to_4h": ["gate_up_proj", "up_proj"],
                "mlp_4h_to_h": "down_proj",
            }

        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir
        config = Phi3Config.from_hugging_face(hf_config_or_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              quant_config=quant_config,
                                              **kwargs)

        if not use_preloading:
            trust_remote_code = kwargs.pop('trust_remote_code', True)

            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, dtype="auto", trust_remote_code=trust_remote_code)

        assert isinstance(hf_model, transformers.PreTrainedModel)

        weights = load_weights_from_hf_model(hf_model, config)

        model = cls(config)
        model.load(weights)
        return model

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
