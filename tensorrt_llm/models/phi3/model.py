import copy
import os
from typing import Optional, Union

import numpy as np
import safetensors
from transformers import AutoModelForCausalLM

from ..._utils import pad_vocab_size
from ...functional import PositionEmbeddingType, Tensor
from ...layers import (MLP, Attention, AttentionMaskType, BlockSparseAttnParams,
                       Embedding, LayerNorm, ParallelLMHead, RmsNorm)
from ...lora_manager import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ...quantization import QuantAlgo
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
        if self.small_variant:
            self.gegelu_limit = config.gegelu_limit

            # MuP uses norm_factor=attention_head_size (rather than sqrt(attention_head_size))
            # We achieve this using q_scaling = sqrt(attention_head_size)
            hidden_size = config.hidden_size
            num_attention_heads = config.num_attention_heads
            attention_head_size = hidden_size / num_attention_heads
            q_scaling = attention_head_size**.5

            block_sparse = (
                (layer_idx + 1) % config.dense_attention_every_n_layers) != 0
            attention_mask_type = AttentionMaskType.blocksparse if block_sparse else AttentionMaskType.causal

            block_sparse_attn_params = BlockSparseAttnParams(
                config.blocksparse_block_size,
                config.blocksparse_homo_head_pattern,
                config.blocksparse_num_local_blocks,
                config.blocksparse_vertical_stride)

            self.input_layernorm = LayerNorm(
                normalized_shape=config.hidden_size, dtype=config.dtype)
            self.post_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                            dtype=config.dtype)
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

            if self.small_variant:
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
            bias=self.small_variant,
            q_scaling=q_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
            rope_scaling_short_factors=rope_scaling_short_factors,
            rope_scaling_long_factors=rope_scaling_long_factors,
            rope_scaling_short_mscale=rope_scaling_short_mscale,
            rope_scaling_long_mscale=rope_scaling_long_mscale,
            original_max_position_embeddings=original_max_position_embeddings,
            block_sparse_params=block_sparse_attn_params)

        self.mlp = MLP(hidden_size=config.hidden_size,
                       ffn_hidden_size=config.intermediate_size,
                       hidden_act=config.hidden_act,
                       dtype=config.dtype,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       quant_mode=config.quant_mode,
                       bias=self.small_variant)

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
        feed_forward_hidden_states = self.mlp(
            post_attention_output,
            gegelu_limit=self.gegelu_limit,
            lora_layer_params=lora_layer_params)
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
        if self.small_variant:
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  dtype=config.dtype)
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

    def __init__(self, config: PretrainedConfig):
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
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, torch_dtype="auto", trust_remote_code=True)

        assert isinstance(hf_model, transformers.PreTrainedModel)

        weights = load_weights_from_hf_model(hf_model, config)

        model = cls(config)
        model.load(weights)
        return model

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'auto',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        device: str = 'cuda',
        calib_dataset: str = 'cnn_dailymail',
        calib_batches: int = 512,
        calib_batch_size: int = 1,
        calib_max_seq_length: int = 512,
        random_seed: int = 1234,
        tokenizer_max_seq_length: int = 2048,
        **kwargs,
    ):
        DEFAULT_MODELOPT_FLOW = [
            QuantAlgo.W4A16_AWQ,
            QuantAlgo.FP8,
            QuantAlgo.W8A8_SQ_PER_CHANNEL,
        ]
        NATIVE_QUANT_FLOW = [QuantAlgo.W4A16, QuantAlgo.W8A16, None]

        config = Phi3Config.from_hugging_face(hf_model_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              quant_config=quant_config,
                                              **kwargs)

        if quant_config.quant_algo in DEFAULT_MODELOPT_FLOW:
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=config.dtype,
                             mapping=config.mapping,
                             quant_config=config.quantization,
                             device=device,
                             calib_dataset=calib_dataset,
                             calib_batches=calib_batches,
                             calib_batch_size=calib_batch_size,
                             calib_max_seq_length=calib_max_seq_length,
                             random_seed=random_seed,
                             tokenizer_max_seq_length=tokenizer_max_seq_length)
        else:
            assert quant_config.quant_algo in NATIVE_QUANT_FLOW, f"Internal error: shall call Modelopt for this quantization {quant_config}"

            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, torch_dtype="auto", trust_remote_code=True)

            for rank in range(mapping.world_size):
                weights = load_weights_from_hf_model(hf_model, config)
                config = copy.deepcopy(config)
                config.set_rank(rank)
                safetensors.torch.save_file(
                    weights, os.path.join(output_dir,
                                          f'rank{rank}.safetensors'))

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
