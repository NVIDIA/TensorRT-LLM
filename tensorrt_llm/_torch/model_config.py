import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

import filelock
import torch
import transformers
from transformers.utils import HF_MODULES_CACHE

from tensorrt_llm import logger
from tensorrt_llm._torch.pyexecutor.config_utils import (is_nemotron_hybrid,
                                                         load_pretrained_config)
from tensorrt_llm._utils import get_sm_version, torch_dtype_to_binding
from tensorrt_llm.bindings import LayerType as LayerTypeCpp
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.llmapi.llm_args import (DeepSeekSparseAttentionConfig,
                                          MoeLoadBalancerConfig)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)


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
    except (PermissionError, filelock.Timeout):
        # Fallback to tempdir
        tmp_dir = Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_lock_path = tmp_dir / "_remote_code.lock"
        tmp_lock = filelock.FileLock(str(tmp_lock_path), timeout=timeout)
        try:
            with tmp_lock:
                yield
        except filelock.Timeout:
            logger.warning(
                f"failed to acquire tempdir config lock within {timeout} seconds, proceeding without lock"
            )
            # proceed without lock
            yield
        except (PermissionError) as e:
            logger.warning(
                f"tempdir config lock unavailable due to OS/permission issue: {e}, proceeding without lock"
            )
            # proceed without lock
            yield


@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    pretrained_config: Optional[TConfig] = None
    mapping: Mapping = field(default_factory=Mapping)

    # Quantization configs
    quant_config: QuantConfig = field(default_factory=QuantConfig)
    # Per linear layer quantization in quant_cfg.json or hf_quant_config.json
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    # Delay weights creation to DecoderModelForCausalLM.__post_init__
    # to support mixed quantization.
    skip_create_weights_in_init: bool = False

    spec_config: Optional["DecodingBaseConfig"] = None
    lora_config: Optional["LoraConfig"] = None
    sparse_attention_config: Optional["SparseAttentionConfig"] = None

    is_generation: bool = True
    max_num_tokens: int = 8192
    max_seq_len: Optional[int] = None

    moe_max_num_tokens: Optional[int] = None
    moe_load_balancer: Optional[MoeLoadBalancerConfig] = None

    attn_backend: str = 'TRTLLM'
    moe_backend: str = 'CUTLASS'  # options can be CUTLASS, TRTLLM
    # IF true, disables FC2+finalize fusion in CUTLASS MoE backend
    moe_disable_finalize_fusion: bool = False
    # If true, use low precision combine in MoE operations (only for NVFP4 quantization)
    use_low_precision_moe_combine: bool = False

    # NVFP4 GEMM backend configuration - list of backends to consider for auto-selection
    # Default excludes 'cutedsl' for faster build time. Add 'cutedsl' for extreme perf.
    nvfp4_gemm_allowed_backends: List[str] = field(
        default_factory=lambda: ['cutlass', 'cublaslt', 'cuda_core'])

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
        and 'pretrained_config' attributes for mutimodal models.
        'quant_config' is allowed to be modified to set different quantization for VLM.
        All the other attributes are frozen.
        This can be bypassed by manually setting '_frozen' to False. The design is
        to discourage modifying the attributes unintentionally.
        """
        if self._frozen:
            if key not in ('_frozen', 'extra_attrs', 'pretrained_config',
                           'quant_config'):
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
                "NCCL_SYMMETRIC": AllReduceStrategy.NCCL_SYMMETRIC,
            }
            key = strategy.upper()
            return maps[key] if key in maps else AllReduceStrategy.AUTO

        if isinstance(self.allreduce_strategy, str):
            self.allreduce_strategy = get_all_reduce_strategy(
                self.allreduce_strategy)

        # Set default moe_max_num_tokens if not specified
        # The maximum number of tokens in MoE are multiplied by DP size when attention DP is enabled
        if self.moe_max_num_tokens is None:
            self.moe_max_num_tokens = self.max_num_tokens * self.mapping.dp_size

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
    def resolve_moe_backend(moe_backend: str, architecture: str) -> str:
        """Resolve AUTO moe_backend to a specific backend based on model architecture.

        Args:
            moe_backend: The configured moe_backend (may be "AUTO")
            architecture: The model architecture name (e.g., "GptOssForCausalLM")

        Returns:
            Resolved backend name (never "AUTO")
        """
        if moe_backend.upper() != "AUTO":
            return moe_backend

        if architecture == "GptOssForCausalLM":
            sm_version = get_sm_version()
            # Select the best performing backend based on SM version
            if 100 <= sm_version < 120:  # Blackwell
                return "TRTLLM"
            elif 90 <= sm_version < 100:  # Hopper
                return "TRITON"
            else:
                return "CUTLASS"  # Fallback to CUTLASS for other SM versions (e.g., SM120)

        return "CUTLASS"

    @staticmethod
    def load_modelopt_quant_config(quant_config_file, checkpoint_dir,
                                   moe_backend):
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
            json_extended_quant_configs: dict = {}
            # See tests/unittest/llmapi/test_llm_quant.py
            try:
                mixed_quant_config_file = transformers.utils.hub.cached_file(
                    checkpoint_dir, 'quant_cfg.json')
                with open(mixed_quant_config_file) as fm:
                    json_extended_quant_configs = json.load(fm)
            except Exception:
                logger.info(
                    f"No quant_cfg.json found for layer quant info, using hf_quant_config.json."
                )
            json_quant_configs.update(json_extended_quant_configs)
            # kv_cache_quant_algo is global regardless of MIXED_PRECISION
            kv_cache_quant_algo = json_quant_configs.get(
                'kv_cache_quant_algo', None)
            mixed_quant_configs = json_quant_configs.get(
                'quantized_layers', None)
            if (kv_quant_lhs := json_extended_quant_configs.get(
                    "kv_cache_quant_algo", None)) is not None and (
                        kv_quant_rhs :=
                        quant_config.kv_cache_quant_algo) is not None:
                if kv_quant_lhs != kv_quant_rhs:
                    raise RuntimeError(
                        f"The kvcache config in 'quant_cfg.json', {kv_quant_lhs},"
                        f"is different from 'hf_quant_config.json', {kv_quant_rhs}!"
                    )
            quant_config.kv_cache_quant_algo = json_quant_configs[
                "kv_cache_quant_algo"]
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

        # Read exclude_modules from HF config if present (HF format module names)
        hf_exclude_modules = hf_quant_config.get('modules_to_not_convert', None)

        # DeepSeek V3 FP8 ckpt
        if hf_quant_config.get("quant_method") == "fp8" and hf_quant_config.get(
                "weight_block_size", []):
            quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES

            block_size = hf_quant_config.get("weight_block_size", [])
            assert tuple(block_size) == (
                128, 128), "FP8_BLOCK_SCALES only supports block_size=(128,128)"
            quant_config.group_size = block_size[0]

            # Set default exclude_modules for FP8_BLOCK_SCALES
            if moe_backend == 'TRTLLM':
                default_exclude = ["*kv_b_proj*", "*k_b_proj*", "*eh_proj"]
            else:
                default_exclude = ["*eh_proj"]

            # Merge HF config's modules_to_not_convert with default exclude_modules
            if hf_exclude_modules is not None:
                quant_config.exclude_modules = list(
                    set(hf_exclude_modules + default_exclude))
            else:
                quant_config.exclude_modules = default_exclude
        # MXFP4 checkpoints.
        elif hf_quant_config.get("quant_method") == "mxfp4":
            quant_config.quant_algo = ModelConfig.get_mxfp4_quant_algo(
                moe_backend)
            quant_config.group_size = 32

            # Default exclude_modules for MXFP4 (TRTLLM internal format)
            default_exclude = [
                'block.*.attn.out', 'block.*.mlp.gate', 'block.*.attn.qkv',
                'embedding', 'unembedding'
            ]

            # Merge HF config's modules_to_not_convert with default exclude_modules
            if hf_exclude_modules is not None:
                quant_config.exclude_modules = list(
                    set(hf_exclude_modules + default_exclude))
            else:
                quant_config.exclude_modules = default_exclude

        # NOTE: This is for llm-compressor's quantized checkpoints.
        elif hf_quant_config.get("quant_method") == "compressed-tensors":
            config_groups = hf_quant_config.get("config_groups")
            if config_groups is None:
                raise ValueError(
                    f"config_groups is not set in {hf_quant_config}.")

            weights_quant_config = config_groups["group_0"]["weights"]
            inputs_quant_config = config_groups["group_0"]["input_activations"]
            weights_quant_strategy = weights_quant_config["strategy"]
            inputs_quant_strategy = inputs_quant_config["strategy"]

            if weights_quant_config["num_bits"] == 8:
                if weights_quant_strategy == "channel":
                    if inputs_quant_strategy != "token":
                        raise ValueError(
                            f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}."
                        )
                    quant_config.quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
                elif weights_quant_strategy == "block":
                    if inputs_quant_strategy != "group":
                        raise ValueError(
                            f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}."
                        )
                    quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                    group_size = inputs_quant_config["group_size"]

                    # NOTE: TRT-LLM only supports group_size=128 for FP8_BLOCK_SCALES.
                    if group_size != 128:
                        raise ValueError(
                            f"Unsupported group_size: {group_size}. Supported: 128."
                        )
                    quant_config.group_size = group_size

                else:
                    raise ValueError(
                        f"Unsupported weights_quant_strategy: {weights_quant_strategy}. "
                        "Supported strategies: 'channel', 'block'.")
            else:
                raise ValueError(
                    f"Unsupported quant_bits: {weights_quant_config['num_bits']}. "
                    "Supported: 8.")

            quant_config.exclude_modules = hf_quant_config.get("ignore", [])
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
            # When handling the case where model_format is TLLM_ENGINE
            # send cyclic requests to the NONE URL.
            if checkpoint_dir is not None:
                pretrained_config = load_pretrained_config(
                    checkpoint_dir,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
                if pretrained_config.architectures[
                        0] == "DeepseekV32ForCausalLM":
                    sparse_attention_config = kwargs.get(
                        'sparse_attention_config')
                    if sparse_attention_config:
                        index_n_heads = sparse_attention_config.index_n_heads or pretrained_config.index_n_heads
                        index_head_dim = sparse_attention_config.index_head_dim or pretrained_config.index_head_dim
                        index_topk = sparse_attention_config.index_topk or pretrained_config.index_topk
                        indexer_max_chunk_size = sparse_attention_config.indexer_max_chunk_size
                        skip_indexer_for_short_seqs = sparse_attention_config.skip_indexer_for_short_seqs
                    else:
                        index_n_heads = pretrained_config.index_n_heads
                        index_head_dim = pretrained_config.index_head_dim
                        index_topk = pretrained_config.index_topk
                        indexer_max_chunk_size = None
                        skip_indexer_for_short_seqs = True
                    kwargs[
                        'sparse_attention_config'] = DeepSeekSparseAttentionConfig(
                            index_n_heads=index_n_heads,
                            index_head_dim=index_head_dim,
                            index_topk=index_topk,
                            indexer_max_chunk_size=indexer_max_chunk_size,
                            skip_indexer_for_short_seqs=
                            skip_indexer_for_short_seqs)
            else:
                raise ValueError(
                    "checkpoint_dir is None. Cannot load model config without a valid checkpoint directory."
                )

        # Get cached file from path or repo id, return None if not exists.
        def cached_file(path_or_repo_id, file_name):
            try:
                return transformers.utils.hub.cached_file(
                    path_or_repo_id, file_name)
            except OSError:
                return None

        # Some checkpoints lack torch_dtype, populate with dtype
        pretrained_config.torch_dtype = getattr(pretrained_config, 'dtype',
                                                None)

        # Apply model_kwargs to override config parameters if provided
        model_kwargs = kwargs.pop('model_kwargs', None)
        if model_kwargs:

            def _recursive_update_config(config: transformers.PretrainedConfig,
                                         update_dict: Dict[str, Any]):
                """
                Recursively update a PretrainedConfig object with values from update_dict.
                Args:
                    config: PretrainedConfig object to update
                    update_dict: Dictionary with values to update in the config
                """
                for key, value_new in update_dict.items():
                    target_value = getattr(config, key, None)

                    # Handle nested PretrainedConfig objects when value is a dict
                    if isinstance(value_new, dict) and isinstance(
                            target_value, transformers.PretrainedConfig):
                        # Recursively update the nested config
                        logger.info(
                            f"Recursively updating nested config: {key}")
                        _recursive_update_config(target_value, value_new)
                    elif (key in ["torch_dtype", "dtype"]
                          and isinstance(value_new, str)
                          and value_new != "auto"):
                        # check special handling of torch_dtype (DEPRECATED!) and dtype keys to ensure we
                        # use the correct torch.dtype object instead of a string.
                        dtype = getattr(torch, value_new)
                        assert isinstance(dtype,
                                          torch.dtype), f"Invalid {dtype=}"
                        setattr(config, key, dtype)
                        logger.info(
                            f"Applied model_kwargs: {key}={dtype} (previous value: {target_value})"
                        )
                    else:
                        # Direct update for simple values
                        setattr(config, key, value_new)
                        logger.info(
                            f"Applied model_kwargs: {key}={value_new} (previous value: {target_value})"
                        )

            _recursive_update_config(pretrained_config, model_kwargs)

        quant_config = QuantConfig()
        layer_quant_config = None
        moe_backend = kwargs.get('moe_backend', 'AUTO')
        # Resolve AUTO to specific backend based on model architecture
        architecture = pretrained_config.architectures[
            0] if pretrained_config.architectures else ""
        moe_backend = cls.resolve_moe_backend(moe_backend, architecture)
        kwargs['moe_backend'] = moe_backend

        # quantized ckpt in modelopt format
        if quant_config_file := cached_file(checkpoint_dir,
                                            'hf_quant_config.json'):
            quant_config, layer_quant_config = cls.load_modelopt_quant_config(
                quant_config_file, checkpoint_dir, moe_backend)
        # quantized ckpt in other formats
        elif hasattr(pretrained_config, "quantization_config"):
            hf_quant_config = pretrained_config.quantization_config
            quant_config, layer_quant_config = cls.load_hf_quant_config(
                hf_quant_config, moe_backend)
        elif quant_config_file := cached_file(checkpoint_dir, 'dtypes.json'):
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

        # Attention DP should not shard attention heads; use attn_tp_size=1 in that case
        # so downstream KV calculations see the full (non-partitioned) head count.
        attn_tp_size = self.mapping.attn_tp_size if not self.mapping.enable_attention_dp else 1
        attn_cp_size = self.mapping.attn_cp_size

        num_heads = self.pretrained_config.num_attention_heads // (
            attn_tp_size * attn_cp_size)

        hidden_size = self.pretrained_config.hidden_size // attn_tp_size
        num_layers = self.pretrained_config.num_hidden_layers
        num_attention_layers = self.get_num_attention_layers()
        if (self.spec_config is not None
                and self.spec_config.spec_dec_mode.is_mtp_one_model()):
            num_layers += self.spec_config.num_nextn_predict_layers
            num_attention_layers += self.spec_config.num_nextn_predict_layers

        model_config_cpp = ModelConfigCpp(
            vocab_size=self.pretrained_config.vocab_size,
            num_layers=num_layers,
            num_attention_layers=num_attention_layers,
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
                kv_heads // (attn_tp_size * attn_cp_size)
                for kv_heads in num_key_value_heads
            ]
            model_config_cpp.num_kv_heads_per_layer = num_kv_heads_per_layer
        else:
            num_kv_heads = num_key_value_heads // (attn_tp_size * attn_cp_size)
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
                    mlp_hidden_size = self._infer_nemotron_ffn_mult(
                    ) // self.mapping.tp_size
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
        head_size = None
        for head_dim_name in head_dim_names:
            if hasattr(self.pretrained_config, head_dim_name):
                value = getattr(self.pretrained_config, head_dim_name)
                if value is not None:
                    head_size = value
                    break

        if head_size is None:
            assert hidden_size % num_heads == 0, (
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
            calculated_head_size = hidden_size // num_heads
            logger.warning(
                f"head_size/head_dim is not set or None, using default value {calculated_head_size}"
            )
            head_size = calculated_head_size

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
        elif hasattr(
                self.pretrained_config, "architectures"
        ) and self.pretrained_config.architectures is not None and self.pretrained_config.architectures[
                0] in ["Qwen3NextForCausalLM"]:
            # Qwen3NextForCausalLM has hybrid attention pattern(1:3 full attention:linear attention),
            # we need to calculate the number of fullattention layers
            return self.pretrained_config.num_hidden_layers // self.pretrained_config.full_attention_interval
        else:
            return self.pretrained_config.num_hidden_layers
