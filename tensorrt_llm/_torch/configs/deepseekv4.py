# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


# Default Engram embedding vocabulary size per n-gram level.
# Derived from the DeepSeek-V3 tokenizer vocab size (129280) scaled by 5.
_DEFAULT_ENGRAM_VOCAB_SIZE = 129280 * 5


# This is a temporary workaround for DeepSeek-V4 model as HF does not support it yet
# TODO: Remove this once HF supports DeepSeek-V4
class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=4096,
        intermediate_size=14336,
        moe_intermediate_size=2048,
        num_hidden_layers=43,
        num_nextn_predict_layers=0,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_shared_experts=1,
        n_routed_experts=256,
        ep_size=1,
        routed_scaling_factor=1.5,
        kv_lora_rank=448,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        v_head_dim=512,
        qk_nope_head_dim=448,
        topk_method="noaux_tc",
        n_group=8,
        topk_group=4,
        num_experts_per_tok=6,
        moe_layer_freq=1,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        hidden_act="silu",
        max_position_embeddings=65536,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        o_groups=8,
        o_lora_rank=1024,
        n_hash_layers=3,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        window_size=128,
        compress_rope_theta=160000,
        compress_ratios=None,
        swiglu_limit=10.0,
        has_engram=False,
        engram_vocab_size=None,
        engram_max_ngram_size=3,
        engram_n_embed_per_ngram=512,
        engram_n_head_per_ngram=8,
        engram_kernel_size=4,
        engram_pad_id=2,
        engram_layer_ids=None,
        engram_seed=0,
        **kwargs,
    ):
        # DeepSeek-V4 HF config uses `sliding_window`, `num_hash_layers`,
        # and `head_dim` for these internal fields.
        # Accept them as aliases for the internal names the rest of the code uses.
        if "sliding_window" in kwargs:
            window_size = kwargs.pop("sliding_window")
        if "num_hash_layers" in kwargs:
            n_hash_layers = kwargs.pop("num_hash_layers")
        if "head_dim" in kwargs:
            v_head_dim = kwargs.pop("head_dim")
        if "score_func" in kwargs:
            scoring_func = kwargs.pop("score_func")

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.window_size = window_size

        # indexer
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk

        # output projection
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank

        # hash layer and mhc
        self.n_hash_layers = n_hash_layers
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        # kv compression
        self.compress_rope_theta = compress_rope_theta
        self.compress_ratios = compress_ratios
        self.swiglu_limit = swiglu_limit

        # Engram configuration
        self.has_engram = has_engram
        self.engram_vocab_size = (
            engram_vocab_size
            if engram_vocab_size is not None
            else [_DEFAULT_ENGRAM_VOCAB_SIZE, _DEFAULT_ENGRAM_VOCAB_SIZE]
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        self.engram_n_head_per_ngram = engram_n_head_per_ngram
        self.engram_kernel_size = engram_kernel_size
        self.engram_pad_id = engram_pad_id
        self.engram_layer_ids = engram_layer_ids if engram_layer_ids is not None else [1, 15]
        self.engram_seed = engram_seed

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
