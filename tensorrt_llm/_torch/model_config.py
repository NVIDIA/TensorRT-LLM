import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar, Union

import filelock
import torch
import transformers

from transformers import PretrainedConfig
from transformers.utils import HF_MODULES_CACHE

from tensorrt_llm import logger
from tensorrt_llm._torch.pyexecutor.config_utils import is_nemotron_hybrid
from tensorrt_llm._utils import get_sm_version, torch_dtype_to_binding
from tensorrt_llm.bindings import LayerType as LayerTypeCpp
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)


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

        # TODO:?
        # if 'rope_alpha' in model_config:
        #     rope_alpha = model_config['rope_alpha']
        # if 'rope_theta' in model_config:
        #     rope_theta = model_config['rope_theta']
        # if 'rope_scaling' in model_config and model_config['rope_scaling']:
        #     rope_scaling = model_config['rope_scaling']
        #     scaling_type = rope_scaling['type'] if 'type' in rope_scaling else 'dynamic'
        #     assert scaling_type == 'dynamic', f"Now only support dynamic rope scaling, but got {scaling_type}"
        #     if 'alpha' in rope_scaling and rope_scaling['alpha']:
        #         scaling_alpha = rope_scaling['alpha']
        #     else:
        #         scaling_alpha = 1.0

        #     rope_base = rope_theta * scaling_alpha ** (size_per_head / (size_per_head - 2))

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


@dataclass
class MoeLoadBalancerConfig:
    num_slots: Optional[int] = None
    initial_global_assignments: Optional[Dict[int,
                                              List[int]]] = field(default=None,
                                                                  repr=False)
    layer_updates_per_iter: int = 0

    ep_rank: Optional[int] = field(default=None, init=False)
    ep_size: Optional[int] = field(default=None, init=False)

    def setup(self, ep_rank: int, ep_size: int) -> None:
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        assert self.num_slots is not None

    @property
    def num_local_slots(self) -> int:
        return self.num_slots // self.ep_size

    @property
    def slot_start(self) -> int:
        return self.ep_rank * self.num_local_slots

    @property
    def slot_end(self) -> int:
        return self.slot_start + self.num_local_slots

    def get_layer_initial_global_assignments(self, layer_idx: int) -> List[int]:
        if self.initial_global_assignments is not None:
            assert layer_idx in self.initial_global_assignments
            assert len(
                self.initial_global_assignments[layer_idx]) == self.num_slots
            return self.initial_global_assignments[layer_idx]
        else:
            return None


@contextlib.contextmanager
def config_file_lock(timeout: int = 10):
    """
    Context manager for file locking when loading pretrained configs.

    This prevents race conditions when multiple processes try to download/load
    the same model configuration simultaneously.

    Args:
        timeout: Maximum time to wait for lock acquisition in seconds
    """
    # Use a single global lock file in HF cache directory
    # This serializes all model loading operations to prevent race conditions
    lock_path = Path(HF_MODULES_CACHE) / "_remote_code.lock"

    # Create and acquire the lock
    lock = filelock.FileLock(str(lock_path), timeout=timeout)

    try:
        with lock:
            yield
    except filelock.Timeout:
        logger.warning(
            f"Failed to acquire config lock within {timeout} seconds, proceeding without lock"
        )
        # Fallback: proceed without locking to avoid blocking indefinitely
        yield


@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    pretrained_config: Optional[TConfig] = None
    mapping: Mapping = field(default_factory=Mapping)

    # quantization configs
    quant_config: QuantConfig = field(default_factory=QuantConfig)
    # TODO(qijun): support per linear layer quantization
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    # Delay weights creation to DecoderModelForCausalLM.__post_init__
    # to support mixed quantization.
    skip_create_weights_in_init: bool = False

    spec_config: Optional["DecodingBaseConfig"] = None
    lora_config: Optional["LoraConfig"] = None

    is_generation: bool = True
    max_num_tokens: int = 8192
    max_seq_len: Optional[int] = None

    moe_max_num_tokens: Optional[int] = None
    moe_load_balancer: Optional[MoeLoadBalancerConfig] = None

    attn_backend: str = 'TRTLLM'
    moe_backend: str = 'CUTLASS'  # options can be CUTLASS, TRTLLM
    # IF true, disables FC2+finalize fusion in CUTLASS MoE backend
    moe_disable_finalize_fusion: bool = False

    allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO

    # If true, enable min-latency mode. Currently only used for Llama4.
    enable_min_latency: bool = False

    # Allow models to select op according to whether CUDA Graphs are used.
    use_cuda_graph: bool = False

    force_dynamic_quantization: bool = False

    # If true, use torch.compile for embedding layers.
    enable_torch_compile_for_embedding = False

    extra_attrs: Dict = field(default_factory=dict, repr=False, init=False)

    _frozen: bool = field(default=False, init=False, repr=False)

    # If true, ONLY the vision encoder part of the full model is loaded/executed.
    mm_encoder_only: bool = False

    def __setattr__(self, key, value):
        """
        Prevent modification of frozen instance attributes.
        However, we allow modification of 'extra_attrs' attributes for torch.compile
        and 'pretrained_config' attributes for mutimodal models. All the other
        attributes are frozen.
        This can be bypassed by manually setting '_frozen' to False. The design is
        to discourage modifying the attributes unintentionally.
        """
        if self._frozen:
            if key not in ('_frozen', 'extra_attrs', 'pretrained_config'):
                raise AttributeError(
                    f"Cannot modify ModelConfig.'{key}' - instance is frozen")
        super().__setattr__(key, value)

    def __post_init__(self):
        if self.pretrained_config and hasattr(self.pretrained_config,
                                              "architectures"):
            self.is_generation = self.is_generation_model(
                self.pretrained_config.architectures,
                mm_encoder_only=self.mm_encoder_only)

        def get_all_reduce_strategy(strategy: str = "AUTO"):
            maps = {
                "AUTO": AllReduceStrategy.AUTO,
                "NCCL": AllReduceStrategy.NCCL,
                "UB": AllReduceStrategy.UB,
                "MINLATENCY": AllReduceStrategy.MIN_LATENCY,
                "ONESHOT": AllReduceStrategy.ONESHOT,
                "TWOSHOT": AllReduceStrategy.TWOSHOT,
                "LOWPRECISION": AllReduceStrategy.LOWPRECISION,
                "MNNVL": AllReduceStrategy.MNNVL,
                "NCCL_SYMMETRIC": AllReduceStrategy.NCCL_SYMMETRIC
            }
            key = strategy.upper()
            return maps[key] if key in maps else AllReduceStrategy.AUTO

        if isinstance(self.allreduce_strategy, str):
            self.allreduce_strategy = get_all_reduce_strategy(
                self.allreduce_strategy)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the torch dtype of the model."""
        # TODO: this is an assumption that a HF model is always in bfloat16
        # We should figure out a better way to handle this if other models
        # start to not report dtype.
        return self.pretrained_config.torch_dtype or torch.bfloat16

    @property
    def fuse_pos_embd(self):
        if self.attn_backend == 'TRTLLM':
            return True
        elif self.attn_backend == 'FLASHINFER':
            return False
        return False

    @property
    def enable_flash_mla(self):
        if self.attn_backend == 'TRTLLM':
            if getattr(self.pretrained_config,
                       "kv_lora_rank", None) and getattr(
                           self.pretrained_config, "qk_rope_head_dim", None):
                head_dim = self.pretrained_config.kv_lora_rank + self.pretrained_config.qk_rope_head_dim
                if head_dim == 576 and torch.cuda.get_device_capability() == (
                        9, 0):
                    return True
        return False

    def get_quant_config(self, name: Optional[str] = None) -> QuantConfig:
        if name is None or self.per_layer_quant_configs is None:
            return self.quant_config

        if name in self.per_layer_quant_configs:
            return self.per_layer_quant_configs[name]

        raise ValueError(f'quant config of {name} is not found')

    @staticmethod
    def is_generation_model(model_architectures: Optional[List[str]],
                            mm_encoder_only: bool = False) -> bool:
        if model_architectures is None:
            logger.warning(
                "Model architectures is None, default to is_generation_model=True"
            )
            return True
        if mm_encoder_only:
            return False
        return model_architectures[0] not in [
            "BertForSequenceClassification", "Qwen2ForProcessRewardModel",
            "Qwen2ForRewardModel", "LlamaForTextEmbedding"
        ]
        # TODO: should be 'not model_type == ModelType.ENCODER_ONLY'
        # once ModelType is used in pytorch flow.

    @staticmethod
    def load_modelopt_quant_config(quant_config_file, model_dir, moe_backend):
        quant_config = QuantConfig()
        layer_quant_config = None

        with open(quant_config_file) as f:
            quant_config_dict = json.load(f)

        json_quant_configs = quant_config_dict['quantization']

        quant_config.quant_algo = json_quant_configs.get('quant_algo', None)
        # fp8_pb_wo from modelopt is the same as FP8_BLOCK_SCALES
        if quant_config.quant_algo == "fp8_pb_wo":
            quant_config.quant_algo = 'FP8_BLOCK_SCALES'
        quant_config.kv_cache_quant_algo = json_quant_configs.get(
            'kv_cache_quant_algo', None)
        quant_config.group_size = json_quant_configs.get('group_size', None)
        quant_config.exclude_modules = json_quant_configs.get(
            'exclude_modules', None)

        if quant_config.quant_algo == QuantAlgo.MIXED_PRECISION:
            mixed_quant_config_file = model_dir / 'quant_cfg.json'
            with open(mixed_quant_config_file) as fm:
                mixed_quant_configs = json.load(fm)
                # kv_cache_quant_algo is global regardless of MIXED_PRECISION
                kv_cache_quant_algo = mixed_quant_configs['kv_cache_quant_algo']
                mixed_quant_configs = mixed_quant_configs['quantized_layers']
                if kv_cache_quant_algo is not None and quant_config.kv_cache_quant_algo is not None:
                    if kv_cache_quant_algo != quant_config.kv_cache_quant_algo:
                        raise RuntimeError(
                            f"The kvcache config in 'quant_cfg.json', {kv_cache_quant_algo},"
                            f"is different from 'hf_quant_config.json', {quant_config.kv_cache_quant_algo}!"
                        )
                kv_cache_quant_algo = kv_cache_quant_algo or quant_config.kv_cache_quant_algo

                for layer in mixed_quant_configs:
                    config = QuantConfig()
                    config.kv_cache_quant_algo = kv_cache_quant_algo
                    config.quant_algo = mixed_quant_configs[layer]['quant_algo']
                    config.group_size = mixed_quant_configs[layer].get(
                        'group_size', None)
                    mixed_quant_configs[layer] = config
            layer_quant_config = mixed_quant_configs
        elif quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            if quant_config.group_size is None:
                quant_config.group_size = 128

        if moe_backend == 'TRTLLM' and quant_config.quant_algo == "FP8_BLOCK_SCALES" and quant_config.exclude_modules is None:
            quant_config.exclude_modules = [
                "*kv_b_proj*", "*k_b_proj*", "*eh_proj"
            ]
        return quant_config, layer_quant_config

    @staticmethod
    def get_mxfp4_quant_algo(moe_backend, is_dynamic_quant=False):
        quant_algo = ModelConfig.override_quant_algo()
        if quant_algo is None and not is_dynamic_quant:
            if get_sm_version() >= 100:
                if moe_backend == 'TRITON':
                    return QuantAlgo.W4A8_MXFP4_FP8
                else:
                    return QuantAlgo.W4A8_MXFP4_MXFP8
            else:
                return QuantAlgo.W4A16_MXFP4
        else:
            return quant_algo

    @staticmethod
    def load_hf_quant_config(hf_quant_config, moe_backend):
        quant_config = QuantConfig()
        layer_quant_config = None

        # DeepSeek V3 FP8 ckpt
        if hf_quant_config.get("quant_method") == "fp8" and hf_quant_config.get(
                "weight_block_size", []):
            quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
            if moe_backend == 'TRTLLM':
                # TODO: This is a hack. Remove after fp8 bmm is integrated.
                quant_config.exclude_modules = [
                    "*kv_b_proj*", "*k_b_proj*", "*eh_proj"
                ]
            else:
                quant_config.exclude_modules = ["*eh_proj"]

            block_size = hf_quant_config.get("weight_block_size", [])
            assert tuple(block_size) == (
                128, 128), "FP8_BLOCK_SCALES only supports block_size=(128,128)"
            quant_config.group_size = block_size[0]
        # MXFP4 checkpoints.
        elif hf_quant_config.get("quant_method") == "mxfp4":
            quant_config.quant_algo = ModelConfig.get_mxfp4_quant_algo(
                moe_backend)
            quant_config.group_size = 32
            quant_config.exclude_modules = [
                'block.*.attn.out', 'block.*.mlp.gate', 'block.*.attn.qkv',
                'embedding', 'unembedding'
            ]

        return quant_config, layer_quant_config

    @staticmethod
    def load_quant_config_from_dtypes_json(dtypes_json_file, moe_backend: str):
        quant_config = QuantConfig()
        layer_quant_config = None

        exclude_modules = set()
        has_mxfp4 = False
        is_dynamic_quant = False
        with open(dtypes_json_file) as f:
            dtypes_json = json.load(f)
            for layer, dtype in dtypes_json.items():
                if layer.endswith("weight"):
                    if dtype == "BF16" or dtype == "FP16":
                        names = layer.split(".")
                        exclude_modules.add('.'.join(names[:-1]))
                    elif dtype == "MXFP4":
                        # This is the path for the fp8 checkpoint which requires dynamic quantization.
                        is_dynamic_quant = True
                        has_mxfp4 = True
                elif layer.endswith("weight.blocks"):
                    scale_name = layer.replace("weight.blocks", "weight.scales")
                    scale_dtype = dtypes_json.get(scale_name, None)
                    assert scale_dtype == "UE8"
                    is_dynamic_quant = False
                    has_mxfp4 = True

        if has_mxfp4:
            quant_config.quant_algo = ModelConfig.get_mxfp4_quant_algo(
                moe_backend, is_dynamic_quant)
            quant_config.group_size = 32
            quant_config.exclude_modules = list(exclude_modules)
            logger.info(f"Setting quant_config: {quant_config}")

        return quant_config, layer_quant_config

    @staticmethod
    def override_quant_algo():
        new_algo = os.environ.get("OVERRIDE_QUANT_ALGO", None)
        supported_algos = {
            "W4A16_MXFP4": QuantAlgo.W4A16_MXFP4,
            "W4A8_MXFP4_MXFP8": QuantAlgo.W4A8_MXFP4_MXFP8,
            "W4A8_MXFP4_FP8": QuantAlgo.W4A8_MXFP4_FP8,
        }
        if new_algo is not None:
            if new_algo.upper() in supported_algos:
                return supported_algos[new_algo.upper()]
            else:
                logger.warning(
                    f"Unsupported quant algo: {new_algo}, supported algos: {supported_algos.keys()}"
                )
        return None

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str,
                        trust_remote_code=False,
                        **kwargs):
        # Use file lock to prevent race conditions when multiple processes
        # try to import/cache the same remote model config file
        with config_file_lock():
            pretrained_config = transformers.AutoConfig.from_pretrained(
                checkpoint_dir,
                trust_remote_code=trust_remote_code,
            )

            # Find the cache path by looking for the config.json file which should be in all
            # huggingface models
            model_dir = Path(
                transformers.utils.hub.cached_file(checkpoint_dir,
                                                   'config.json')).parent

        quant_config = QuantConfig()
        layer_quant_config = None
        moe_backend = kwargs.get('moe_backend', 'CUTLASS')

        # quantized ckpt in modelopt format
        if (quant_config_file := model_dir / 'hf_quant_config.json').exists():
            quant_config, layer_quant_config = cls.load_modelopt_quant_config(
                quant_config_file, model_dir, moe_backend)
        # quantized ckpt in other formats
        elif hasattr(pretrained_config, "quantization_config"):
            hf_quant_config = pretrained_config.quantization_config
            quant_config, layer_quant_config = cls.load_hf_quant_config(
                hf_quant_config, moe_backend)
        elif (quant_config_file := model_dir / 'dtypes.json').exists():
            quant_config, layer_quant_config = cls.load_quant_config_from_dtypes_json(
                quant_config_file, moe_backend)

        model_config = cls(pretrained_config=pretrained_config,
                           quant_config=quant_config,
                           quant_config_dict=layer_quant_config,
                           **kwargs)
        model_config._frozen = True
        return model_config

    def get_bindings_model_config(self,
                                  tokens_per_block: Optional[int] = None
                                  ) -> "ModelConfigCpp":
        """
        This method is used to construct the bindings config for the model.
        Currently it adheres to gptJsonConfig.cpp::createModelConfig, which assumes
        that an engine has been created.

        Args:
            tokens_per_block: The number of tokens per block. Please note that in PyTorch flow tokens_per_block is not available in the model config, instead it is defined in the executor config.

        Returns:
            The bindings model config.
        """
        # TODO smor- this isn't robust, and currently tested for LlamaConfig only
        # TODO smor- currently assuming no rnn layers, no MOE
        from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp

        num_heads = self.pretrained_config.num_attention_heads // (
            self.mapping.tp_size * self.mapping.cp_size)

        hidden_size = self.pretrained_config.hidden_size // self.mapping.tp_size

        model_config_cpp = ModelConfigCpp(
            vocab_size=self.pretrained_config.vocab_size,
            num_layers=self.pretrained_config.num_hidden_layers,
            num_attention_layers=self.get_num_attention_layers(),
            num_rnn_layers=0,
            num_heads=num_heads,
            hidden_size=hidden_size,
            data_type=torch_dtype_to_binding(
                self.pretrained_config.torch_dtype))

        # For kv cache size calculation: set tokens_per_block
        if tokens_per_block is None:
            logger.warning(
                f"tokens_per_block is not set, using default value {model_config_cpp.tokens_per_block}"
            )
        else:
            model_config_cpp.tokens_per_block = tokens_per_block

        num_key_value_heads = getattr(self.pretrained_config,
                                      "num_key_value_heads", num_heads)
        if isinstance(num_key_value_heads, (list, tuple)):
            # Per-layer KV heads (e.g., Nemotron-NAS, variable GQA models)
            num_kv_heads_per_layer = [
                kv_heads // (self.mapping.tp_size * self.mapping.cp_size)
                for kv_heads in num_key_value_heads
            ]
            model_config_cpp.num_kv_heads_per_layer = num_kv_heads_per_layer
        else:
            num_kv_heads = num_key_value_heads // (self.mapping.tp_size *
                                                   self.mapping.cp_size)
            model_config_cpp.set_num_kv_heads(num_kv_heads)

        mlp_hidden_size = None
        if self.pretrained_config.intermediate_size is not None:
            mlp_hidden_size = self.pretrained_config.intermediate_size // self.mapping.tp_size
        else:
            # TODO: once tensorrt_llm._torch.AutoConfig is implemented, the following logic
            # should be moved to tensorrt_llm._torch.AutoConfig of the relevant modeling_xxx file
            if hasattr(self.pretrained_config, "architectures"
                       ) and self.pretrained_config.architectures is not None:
                architectures = self.pretrained_config.architectures
                if len(architectures
                       ) == 1 and architectures[0] == "DeciLMForCausalLM":
                    mlp_hidden_size = self._infer_nemotron_ffn_mult()
                else:
                    raise ValueError(
                        f"Inferring mlp hidden size for model architecture: {architectures} isn't supported yet"
                    )
        if mlp_hidden_size is None:
            raise ValueError(
                f"Failed to infer mlp hidden size for model: {self.pretrained_config.model_type}"
            )

        # For kv cache size calculation: set size_per_head
        head_dim_names = ["head_size", "head_dim"]
        for head_dim_name in head_dim_names:
            if head_dim_name in self.pretrained_config:
                head_size = getattr(self.pretrained_config, head_dim_name)
                break
        else:
            logger.warning(
                f"head_size/head_dim is not set, using default value {hidden_size // num_heads}"
            )
            head_size = hidden_size // num_heads

        model_config_cpp.mlp_hidden_size = mlp_hidden_size
        model_config_cpp.size_per_head = head_size

        # NOTE: this method is not robust, for Gemma3ForCausalLM only
        layer_types = self.get_layer_types()
        if layer_types is not None:
            model_config_cpp.layer_types = layer_types

        return model_config_cpp

    def _infer_nemotron_ffn_mult(self):
        # TODO smor: this is a hack to support Nemotron-Super-49B-v1 with LoRA, tracked by TRTLLM-5045 ticket
        # Nemotron-NAS has variable ffn_mult for each layer, we need to find the maximum
        # so that we don't set a too small mlp_hidden_size. This solution leads to a memory
        # consumption that is higher than required.
        biggest_ffn_mult = max([
            (x.ffn.ffn_mult if x.ffn.ffn_mult is not None else 0)
            for x in self.pretrained_config.block_configs
        ])

        from tensorrt_llm._torch.models.modeling_nemotron_nas import \
            _ffn_mult_to_intermediate_size
        mlp_hidden_size = _ffn_mult_to_intermediate_size(
            biggest_ffn_mult, self.pretrained_config.hidden_size)

        return mlp_hidden_size

    def get_layer_types(self) -> Optional[List[LayerTypeCpp]]:
        """
        This method is a hack to support the effort to switch to KvCacheManagerCpp.
        Currently, it is only tested for Gemma3ForCausalLM. For other models, it will return None.
        """
        if self.pretrained_config.architectures[0] in ["Gemma3ForCausalLM"]:
            logger.debug(
                f"Setting layer types for {self.pretrained_config.architectures}"
            )
            return [
                LayerTypeCpp.ATTENTION,
            ] * self.pretrained_config.num_hidden_layers
        else:
            return None

    def get_num_attention_layers(self):
        if is_nemotron_hybrid(self.pretrained_config):
            return self.pretrained_config.hybrid_override_pattern.count("*")
        else:
            return self.pretrained_config.num_hidden_layers
