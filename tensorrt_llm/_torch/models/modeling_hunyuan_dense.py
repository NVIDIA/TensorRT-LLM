from typing import Dict, List, Optional, Union

import torch
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             duplicate_kv_weight, register_auto_model)


class HunYuanPretrainedConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HunYuanModel`]. It is used to instantiate an
    HunYuan model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the HunYuan-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the HunYuan model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HunYuanModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations or shared MLP representations.
        moe_intermediate_size (`int` or `List`, *optional*, defaults to 11008):
            Dimension of the MLP representations in MoE. Use a list if you want a different size per layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether query and key in attention use norm
        use_cla (`bool`, *optional*, defaults to `False`):
            Whether to use CLA in attention
        cla_share_factor (`int`, *optional*, defaults to 1):
            The share factor of CLA
        num_experts (`int` or `List`, *optional*, defaults to 1):
            The number of experts for moe. If it is a list, it will be used as the number of experts for each layer.
        num_shared_expert (`int` or `List`, *optional*, defaults to 1):
            The number of shared experts for moe. If it is a list, it will be used as the number of shared experts for each layer.
        moe_topk (`int` or `List`, *optional*, defaults to 1):
            The topk value for moe. If it is a list, it will be used as the topk value for each layer.
        capacity_factor (Not used) (`float` or `List`, *optional*, defaults to 1.0):
            The capacity factor for moe. If it is a list, it will be used as the capacity factor for each layer.
        moe_layer_num_skipped (`int`, *optional*, defaults to 0):
            First moe_layer_num_skipped layers do not use MoE.
        mamba (`bool`, *optional*, defaults to `False`):
            Whether to use hybrid mamba mode.
        hybrid_override_pattern (`str`, *optional*, defaults to None):
            The hybrid pattern when using hybrid mamba mode. "*" represents self-attention, "-" represents mlp, "M" represents Mamba layer.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for mamba layer output.
    """

    model_type = "hunyuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=290943,
        org_vocab_size=290943,
        hidden_size=4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: Union[int, List] = None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        attention_head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eod_token_id=3,
        im_start_id=4,
        im_end_id=5,
        text_start_id=6,
        text_end_id=7,
        image_token_id=8,
        video_start_id=9,
        video_end_id=10,
        im_newline_id=11,
        mask_init_id=12,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        use_qk_norm=False,
        use_rotary_pos_emb=True,
        use_cla=False,
        cla_share_factor=1,
        norm_type="hf_rms",
        num_experts: Union[int, List] = 1,
        use_mixed_mlp_moe=False,
        num_shared_expert: Union[int, List] = 1,
        moe_topk: Union[int, List] = 1,
        # capacity_factor: Union[int, List]=1.0,
        moe_drop_tokens=False,
        moe_random_routing_dropped_token=False,
        use_mla=False,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        moe_layer_num_skipped=0,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        group_limited_greedy=False,
        n_group=None,
        topk_group=None,
        mamba=False,
        hybrid_override_pattern=None,
        mamba_ssm_ngroups=8,
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_expand=4,
        mamba_head_dim=64,
        mamba_use_seq_idx=False,
        hidden_dropout=0.0,
        vit_path=None,
        num_media_embeds=257,
        vit_type="AnyResVit",
        vit_input_resolution=224,
        vit_token=64,
        vit_patch=1,
        vit_mapping_type="simple_conv_mlp",
        vit_norm_type="fused",
        vit_used_rms_norm=True,
        vit_remove_prenorm=True,
        vit_add_patchemb_bias=True,
        anyres_vit_max_image_size=2048,
        anyres_pooling_size=2,
        anyres_vit_two_views=False,
        skip_cls_token=False,
        position_embedding_xdrope=False,
        xdrope_section=None,
        add_classification_head=False,
        class_num=0,
        pool_type="last",
        pad_id=-1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.org_vocab_size = org_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        # self.capacity_factor = capacity_factor
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_random_routing_dropped_token = moe_random_routing_dropped_token

        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        else:
            self.attention_head_dim = self.hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # self._rope_scaling_validation()   # TODO: Need validation?
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.use_cla = use_cla
        self.cla_share_factor = cla_share_factor
        self.norm_type = norm_type
        # MLA args
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # DeepSeek related args
        self.moe_layer_num_skipped = moe_layer_num_skipped
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.group_limited_greedy = group_limited_greedy
        self.n_group = n_group
        self.topk_group = topk_group
        self.add_classification_head = add_classification_head
        self.class_num = class_num
        self.pool_type = pool_type
        self.pad_id = pad_id

        if self.class_num is not None:
            self.dense_list = [self.hidden_size, self.class_num]

        # Vit args
        self.vit_path = vit_path
        self.num_media_embeds = num_media_embeds
        self.vit_type = vit_type
        self.vit_input_resolution = vit_input_resolution
        self.vit_token = vit_token
        self.vit_patch = vit_patch
        self.vit_mapping_type = vit_mapping_type
        self.vit_norm_type = vit_norm_type
        self.vit_used_rms_norm = vit_used_rms_norm
        self.vit_remove_prenorm = vit_remove_prenorm
        self.vit_add_patchemb_bias = vit_add_patchemb_bias
        self.anyres_vit_max_image_size = anyres_vit_max_image_size
        self.anyres_pooling_size = anyres_pooling_size
        self.anyres_vit_two_views = anyres_vit_two_views
        self.skip_cls_token = skip_cls_token
        self.position_embedding_xdrope = position_embedding_xdrope
        self.xdrope_section = xdrope_section

        # token id
        self.eod_token_id = eod_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.text_start_id = text_start_id
        self.text_end_id = text_end_id
        self.image_token_id = image_token_id
        self.video_start_id = video_start_id
        self.video_end_id = video_end_id
        self.im_newline_id = im_newline_id
        self.mask_init_id = mask_init_id

        # Mamba args
        self.mamba = mamba
        self.hybrid_override_pattern = hybrid_override_pattern
        self.mamba_ssm_ngroups = mamba_ssm_ngroups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_head_dim = mamba_head_dim
        self.mamba_use_seq_idx = mamba_use_seq_idx
        self.chunk_size = 256
        assert not (self.mamba
                    and self.use_cla), "Mamba and CLA are mutually exclusive."

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(
                self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor` or `type` and `alpha`, "
                f"got {self.rope_scaling}")
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        rope_scaling_alpha = self.rope_scaling.get("alpha", None)
        if rope_scaling_type is None or rope_scaling_type not in [
                "linear", "dynamic"
        ]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None and rope_scaling_alpha is None:
            raise ValueError(
                "`rope_scaling`'s factor or alpha field must be have one, got both of none"
            )
        if rope_scaling_factor is not None:
            if not isinstance(rope_scaling_factor,
                              float) or rope_scaling_factor <= 1.0:
                raise ValueError(
                    f"`rope_scaling`'s factor field must be a float > 1.0, got {rope_scaling_factor}"
                )
        if rope_scaling_alpha is not None:
            if not isinstance(rope_scaling_alpha,
                              float) or rope_scaling_alpha <= 1.0:
                raise ValueError(
                    f"`rope_scaling`'s alpha field must be a float > 1.0, got {rope_scaling_alpha}"
                )


class HunYuanTRTRotaryEmbedding(RotaryEmbedding):

    def __init__(self,
                 config: HunYuanPretrainedConfig,
                 device: Optional[torch.device] = None):
        super().__init__(
            config,
            head_dim=config.hidden_size // config.num_attention_heads,
        )


class HunYuanAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[HunYuanPretrainedConfig],
        layer_idx: Optional[int] = None,
        nope_layer: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config

        self.use_rope = not nope_layer
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(config),
            is_neox=True,
        ) if self.use_rope else None

        self.use_qk_norm = config.use_qk_norm
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            # rotary_emb=HunYuanTRTRotaryEmbedding(config),
            # pos_embd_params=None,
            pos_embd_params=pos_embd_params,
            rope_fusion=not self.use_qk_norm,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )
        if self.use_qk_norm:
            if config.head_dim != None:
                self.head_dim = config.head_dim
            else:
                self.head_dim = config.hidden_size // config.num_attention_heads
            self.query_layernorm = RMSNorm(hidden_size=self.head_dim,
                                           eps=config.rms_norm_eps,
                                           dtype=config.torch_dtype)
            self.key_layernorm = RMSNorm(hidden_size=self.head_dim,
                                         eps=config.rms_norm_eps,
                                         dtype=config.torch_dtype)
            self.aux_stream = aux_stream
            self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        q, k, v = self.split_qkv(q, k, v)
        if position_ids is not None:
            q, k, v = super().apply_rope(q, k, v, position_ids)
        # Llama4 applies QK norm after RoPE.
        if self.use_qk_norm:
            q, k = self.apply_qk_norm(q, k)

        return q, k, v

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.query_layernorm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.key_layernorm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert lora_params is None, "LORA is not supported for HunYuanAttention"
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            mrope_config=mrope_config,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            **kwargs,
        )


class HunYuanDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[HunYuanPretrainedConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        # attention
        self.self_attn = HunYuanAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            bias=config.mlp_bias,
                            dtype=config.torch_dtype,
                            config=model_config)

        norm_type = getattr(config, 'norm_type', 'rms')
        if norm_type == 'hf_rms' or norm_type == 'rms':
            self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                           eps=config.rms_norm_eps,
                                           dtype=config.torch_dtype)
            self.post_attention_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype)
        elif norm_type == 'fused' or norm_type == 'torch_nn':
            self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        # Fully Connected
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HunYuanModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[HunYuanPretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.aux_stream_dict = {
            key: torch.cuda.Stream()
            for key in [
                AuxStreamType.Attention, AuxStreamType.MoeShared,
                AuxStreamType.MoeChunkingOverlap
            ]
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList([
            HunYuanDecoderLayer(model_config, layer_idx, self.aux_stream_dict)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            kwargs['layer_idx'] = layer_idx
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("HunYuanDenseV1ForCausalLM")
class HunYuanDenseV1ForCausalLM(DecoderModelForCausalLM[HunYuanModel,
                                                        HunYuanPretrainedConfig]
                                ):

    def __init__(self, model_config: ModelConfig[HunYuanPretrainedConfig]):
        super().__init__(HunYuanModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
        self._execution_stats = None

    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads)

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

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
                names = name.split('.')
                if names[-1] in params_map:
                    # model.layers.{idx}.mlp.shared_mlp.gate_up_proj or model.layers.{idx}.self_attn.qkv_proj
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    num_kv_heads=v[:].shape[0] // head_dim,
                                    tensor_parallel_size=tp_size)
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    name = name.replace('gate', 'gate.wg')
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        # model.layers.{idx}.self_attn.o_proj or model.layers.{idx}.mlp.shared_mlp.down_proj
                        # or model.layers.{idx}.mlp.experts.gate
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
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
