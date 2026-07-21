# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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

import contextlib
import errno
import json
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar

import filelock
import torch
import transformers
from transformers.utils import HF_MODULES_CACHE

from tensorrt_llm._torch.pyexecutor.config_utils import (
    get_qwen3_hybrid_num_attention_layers, is_nemotron_hybrid, is_qwen3_hybrid,
    load_pretrained_config)
from tensorrt_llm._utils import (get_sm_version, is_sm_100f,
                                 torch_dtype_to_binding)
from tensorrt_llm.bindings import LayerType as LayerTypeCpp
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.llmapi.llm_args import (DeepSeekSparseAttentionConfig,
                                          DeepSeekV4SparseAttentionConfig,
                                          KvCacheConfig, MoeLoadBalancerConfig,
                                          MultimodalConfig)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.quant_config_utils import \
    update_quant_config_from_compressed_tensors
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.modelopt_config import (
    is_modelopt_quant_config, read_modelopt_quant_config,
    warn_if_inline_diverges)

if TYPE_CHECKING:
    from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
    from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig, LoraConfig,
                                              SparseAttentionConfig,
                                              SpeculativeConfig)

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)

_DEEPSEEK_V4_ARCHITECTURES = {"DeepseekV4ForCausalLM"}
_DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT = "layers.0.ffn.experts.0.w1.weight"

_MINIMAX_M3_ARCHITECTURES = {
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
}


def _is_lock_infra_error(exc: BaseException) -> bool:
    """Whether exc indicates broken lock infrastructure (not mere contention)."""
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        return exc.errno in (errno.EACCES, errno.EPERM, errno.ENOLCK,
                             errno.ESTALE)
    return False


@contextlib.contextmanager
def config_file_lock(timeout: int = 10):
    """
    Context manager for file locking when loading pretrained configs.

    This prevents race conditions when multiple processes try to download/load
    the same model configuration simultaneously.

    Args:
        timeout: Maximum time to wait for lock acquisition in seconds
    """
    # Use a single global lock file in HF cache directory to serialize loads.
    lock_path = Path(HF_MODULES_CACHE) / "_remote_code.lock"
    lock = filelock.FileLock(str(lock_path), timeout=timeout)

    # Guard only acquisition so caller-body exceptions propagate (single-yield).
    try:
        lock.acquire(timeout=timeout)
    except filelock.Timeout:
        # Contention, not broken infra: a tempdir lock can't serialize against
        # the holder, so degrade to no lock instead of crashing the process.
        logger.warning(
            f"could not acquire config lock within {timeout}s, proceeding without lock"
        )
        yield
    except (PermissionError, OSError) as e:
        # Broken lock infra (perms / NFS ENOLCK/ESTALE): retry on a tempdir lock.
        if not _is_lock_infra_error(e):
            raise
        tmp_dir = Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_lock = filelock.FileLock(str(tmp_dir / "_remote_code.lock"),
                                     timeout=timeout)
        try:
            tmp_lock.acquire(timeout=timeout)
        except (PermissionError, OSError, filelock.Timeout):
            logger.warning(
                "tempdir config lock unavailable, proceeding without lock")
            yield
        else:
            try:
                yield
            finally:
                tmp_lock.release()
    else:
        try:
            yield
        finally:
            lock.release()


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
    is_encoder_decoder: bool = False
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

    # cute dsl op configs
    use_cute_dsl_blockscaling_mm: bool = False
    use_cute_dsl_blockscaling_bmm: bool = False
    use_cute_dsl_bf16_bmm: bool = False
    use_cute_dsl_bf16_gemm: bool = False

    _frozen: bool = field(default=False, init=False, repr=False)

    # If true, ONLY the vision encoder part of the full model is loaded/executed.
    mm_encoder_only: bool = False

    # If true, the multimodal encoder of a multimodal checkpoint is NOT
    # instantiated/loaded and the model serves text-only requests. This is
    # opt-in per model: each model implementation must honor this flag when
    # building its encoder (currently the Qwen3-VL / Qwen3.5-VL models); a
    # model that does not check it simply ignores the flag (no-op).
    disable_mm_encoder: bool = False

    # Video pruning rate for VLM models (None = EVS disabled)
    video_pruning_rate: Optional[float] = None

    # Multimodal model configuration, e.g. vision encoder CUDA graph buckets.
    multimodal_config: MultimodalConfig | None = None

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
        if self.pretrained_config:
            self.is_encoder_decoder = self.is_encoder_decoder_model(
                self.pretrained_config)

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
    def is_encoder_decoder_model(pretrained_config: Optional[TConfig]) -> bool:
        if pretrained_config is None:
            return False
        text_config = pretrained_config
        get_text_config = getattr(pretrained_config, "get_text_config", None)
        if callable(get_text_config):
            text_config = get_text_config()
        elif hasattr(pretrained_config, "text_config"):
            text_config = pretrained_config.text_config
        return getattr(text_config, "is_encoder_decoder", False)

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
            "Qwen2ForRewardModel", "LlamaForTextEmbedding",
            "Qwen3ForTextEmbedding"
        ]
        # TODO: should be 'not model_type == ModelType.ENCODER_ONLY'
        # once ModelType is used in pytorch flow.

    @staticmethod
    def resolve_moe_backend(moe_backend: str,
                            architecture: str,
                            quant_config: Optional[QuantConfig] = None) -> str:
        """Resolve AUTO moe_backend to a specific backend based on model architecture.

        Args:
            moe_backend: The configured moe_backend (may be "AUTO")
            architecture: The model architecture name (e.g., "GptOssForCausalLM")
            quant_config: Optional quantization config for resolving quantized
                MoE checkpoints.

        Returns:
            Resolved backend name (never "AUTO")
        """
        if moe_backend.upper() != "AUTO":
            return moe_backend

        if architecture in _DEEPSEEK_V4_ARCHITECTURES:
            sm_version = get_sm_version()
            if 100 <= sm_version < 120:
                return "TRTLLM"

        if architecture == "GptOssForCausalLM":
            sm_version = get_sm_version()
            # Select the best performing backend based on SM version
            if 100 <= sm_version < 120:  # Blackwell
                return "TRTLLM"
            elif 90 <= sm_version < 100:  # Hopper
                return "TRITON"
            else:
                return "CUTLASS"  # Fallback to CUTLASS for other SM versions (e.g., SM120)

        quant_algo = quant_config.quant_algo if quant_config is not None else None
        is_fp8_block_scales = quant_algo in (QuantAlgo.FP8_BLOCK_SCALES,
                                             "FP8_BLOCK_SCALES")
        if is_fp8_block_scales and is_sm_100f():
            return "TRTLLM"

        return "CUTLASS"

    @staticmethod
    def load_modelopt_quant_config(quant_config_file, checkpoint_dir,
                                   moe_backend):
        with open(quant_config_file) as f:
            quant_config_dict = json.load(f)
        return ModelConfig._build_modelopt_quant_config(
            read_modelopt_quant_config(quant_config_dict), checkpoint_dir,
            moe_backend)

    @staticmethod
    def _build_modelopt_quant_config(json_quant_configs, checkpoint_dir,
                                     moe_backend):
        """Build (quant_config, layer_quant_config) from a normalized modelopt 'quantization' inner dict.

        ``json_quant_configs`` should be a dict as produced by
        :func:`read_modelopt_quant_config`. May be mutated in place via
        ``.update()`` when overlaying ``quant_cfg.json``.
        """
        quant_config = QuantConfig()
        layer_quant_config = None

        def _algo_or_none(value):
            # modelopt hf_quant_config.json may spell "no quantization" as JSON
            # null (-> None) OR as the string "none"/"null" (e.g. the Inkling
            # NVFP4 checkpoint uses ``"kv_cache_quant_algo": "none"``); both must
            # map to None rather than QuantAlgo("none"), which is not a member.
            if value is None or (isinstance(value, str)
                                 and value.strip().lower() in ("none", "null",
                                                               "")):
                return None
            return QuantAlgo(value)

        quant_config.quant_algo = _algo_or_none(
            json_quant_configs.get('quant_algo'))
        quant_config.kv_cache_quant_algo = _algo_or_none(
            json_quant_configs.get('kv_cache_quant_algo'))
        quant_config.group_size = json_quant_configs.get('group_size', None)
        quant_config.exclude_modules = json_quant_configs.get(
            'exclude_modules', None)
        # AWQ-specific extras; only override defaults when present in JSON.
        if 'has_zero_point' in json_quant_configs:
            quant_config.has_zero_point = json_quant_configs['has_zero_point']
        if 'pre_quant_scale' in json_quant_configs:
            quant_config.pre_quant_scale = json_quant_configs['pre_quant_scale']

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
                    "No quant_cfg.json found for layer quant info, using hf_quant_config.json."
                )
            json_quant_configs.update(json_extended_quant_configs)
            # kv_cache_quant_algo is global regardless of MIXED_PRECISION
            kv_cache_quant_algo = (QuantAlgo(
                json_quant_configs['kv_cache_quant_algo']) if
                                   json_quant_configs.get('kv_cache_quant_algo')
                                   is not None else None)
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
            quant_config.kv_cache_quant_algo = kv_cache_quant_algo
            quant_config.group_size = json_quant_configs.get(
                'group_size', quant_config.group_size)
            quant_config.exclude_modules = json_quant_configs.get(
                'exclude_modules', quant_config.exclude_modules)

            for layer in mixed_quant_configs:
                layer_cfg = mixed_quant_configs[layer]
                config = QuantConfig()
                config.kv_cache_quant_algo = kv_cache_quant_algo
                config.quant_algo = QuantAlgo(layer_cfg['quant_algo'])
                config.group_size = layer_cfg.get('group_size', None)
                # AWQ-specific extras emitted by modelopt per-layer.
                if 'has_zero_point' in layer_cfg:
                    config.has_zero_point = layer_cfg['has_zero_point']
                if 'pre_quant_scale' in layer_cfg:
                    config.pre_quant_scale = layer_cfg['pre_quant_scale']
                mixed_quant_configs[layer] = config
            layer_quant_config = mixed_quant_configs
        elif quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            if quant_config.group_size is None:
                quant_config.group_size = 128

        if (moe_backend == 'TRTLLM'
                and quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
                and quant_config.exclude_modules is None):
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
    def load_hf_quant_config(hf_quant_config, moe_backend, checkpoint_dir=None):
        quant_config = QuantConfig()
        layer_quant_config = None

        # Route inline modelopt configs (legacy or flat) to the modelopt builder.
        if is_modelopt_quant_config(hf_quant_config):
            return ModelConfig._build_modelopt_quant_config(
                read_modelopt_quant_config(hf_quant_config), checkpoint_dir,
                moe_backend)

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
            # kv_b_proj must always be excluded: FP8 128x128 block boundaries
            # don't necessarily align with per-head dim boundaries (e.g. GLM-5
            # has qk_nope_head_dim=192), so the scale tensor cannot be cleanly
            # reshaped per-head. The dequant path handles this correctly.
            default_exclude = ["*kv_b_proj*", "*k_b_proj*", "*eh_proj"]

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

        # MXFP8 checkpoints (e4m3 weights + UE8M0 1x32 block scales, dynamic MXFP8 acts).
        elif hf_quant_config.get("quant_method") == "mxfp8":
            quant_config.quant_algo = QuantAlgo.MXFP8
            block_size = hf_quant_config.get("weight_block_size", [1, 32])
            # MXFP8 uses 1x32 blocks along the K dim; group_size is the K block (32).
            assert tuple(block_size) == (1, 32), (
                f"MXFP8 only supports weight_block_size=[1,32], got {block_size}"
            )
            quant_config.group_size = block_size[1]

            # Layers the producer left in BF16.
            ignored = hf_quant_config.get("ignored_layers", [])
            if hf_exclude_modules is not None:
                quant_config.exclude_modules = list(
                    dict.fromkeys(hf_exclude_modules + ignored))
            else:
                quant_config.exclude_modules = list(ignored)

        # NOTE: This is for llm-compressor's quantized checkpoints.
        elif hf_quant_config.get("quant_method") == "compressed-tensors":
            update_quant_config_from_compressed_tensors(quant_config,
                                                        hf_quant_config)
        elif hf_quant_config.get("quant_method") == "nvfp4":
            quant_config.quant_algo = QuantAlgo.NVFP4
            group_size = hf_quant_config.get("group_size", 16)
            assert group_size == 16, "NVFP4 only supports group_size=16"
            quant_config.group_size = group_size
            default_exclude = ['*.mlp.gate', 'lm_head']

            # Merge HF config's modules_to_not_convert with default exclude_modules
            if hf_exclude_modules is not None:
                quant_config.exclude_modules = list(
                    dict.fromkeys(hf_exclude_modules + default_exclude))
            else:
                quant_config.exclude_modules = default_exclude
        return quant_config, layer_quant_config

    @staticmethod
    def _read_safetensors_header(path: Path) -> Dict[str, Any]:
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(header_size))

    @staticmethod
    def _get_safetensors_header_for_tensor(checkpoint_dir: str,
                                           tensor_name: str) -> Optional[Dict]:
        checkpoint_path = Path(checkpoint_dir)
        candidates = []
        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            shard_name = index.get("weight_map", {}).get(tensor_name)
            if shard_name is not None:
                candidates.append(checkpoint_path / shard_name)

        candidates.extend(sorted(checkpoint_path.glob("*.safetensors")))
        seen = set()
        for candidate in candidates:
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)
            header = ModelConfig._read_safetensors_header(candidate)
            if tensor_name in header:
                return header[tensor_name]
        return None

    @staticmethod
    def _detect_deepseek_v4_routed_moe_layout(
            checkpoint_dir: str) -> Optional[str]:
        tensor_info = ModelConfig._get_safetensors_header_for_tensor(
            checkpoint_dir, _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT)
        if tensor_info is None:
            return None

        dtype = tensor_info.get("dtype")
        shape = tensor_info.get("shape", [])
        if dtype == "I8" and len(shape) == 2:
            return "mxfp4"
        if dtype == "U8":
            return "nvfp4"
        return None

    @staticmethod
    def _is_deepseek_v4_base_checkpoint(checkpoint_dir: str) -> bool:
        tensor_info = ModelConfig._get_safetensors_header_for_tensor(
            checkpoint_dir, _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT)
        if tensor_info is None:
            return False

        return ModelConfig._detect_deepseek_v4_routed_moe_layout(
            checkpoint_dir) not in ("mxfp4", "nvfp4")

    @staticmethod
    def _has_deepseek_v4_layer_only_modelopt_quant_config(
            quant_config_file: str) -> bool:
        with open(quant_config_file) as f:
            quant_config_dict = json.load(f)

        quantization_config = quant_config_dict.get('quantization', {})
        return (quantization_config.get('quant_algo', None) is None and
                quantization_config.get('quantized_layers', None) is not None)

    @staticmethod
    def _set_deepseek_v4_routed_moe_quant_config(pretrained_config,
                                                 checkpoint_dir: str,
                                                 moe_backend: str,
                                                 layer_quant_config,
                                                 spec_config=None,
                                                 require_layout: bool = False):
        layout = ModelConfig._detect_deepseek_v4_routed_moe_layout(
            checkpoint_dir)
        if layout not in ("mxfp4", "nvfp4"):
            if require_layout:
                raise ValueError(
                    "DeepSeek-V4 checkpoint has layer-specific quantized_layers "
                    "in hf_quant_config.json, but the routed MoE layout could "
                    "not be detected from safetensors metadata. Expected "
                    f"{_DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT} to use dtype I8 "
                    "for MXFP4 or U8 for NVFP4.")
            return layer_quant_config

        experts_quant_config = QuantConfig()
        if layout == "mxfp4":
            experts_quant_config.quant_algo = ModelConfig.get_mxfp4_quant_algo(
                moe_backend)
            experts_quant_config.group_size = 32
        else:
            experts_quant_config.quant_algo = QuantAlgo.NVFP4
            experts_quant_config.group_size = 16
        experts_quant_config.exclude_modules = [
            'block.*.attn.out', 'block.*.mlp.gate', 'block.*.attn.qkv',
            'embedding', 'unembedding'
        ]

        if layer_quant_config is None:
            layer_quant_config = {}
        else:
            layer_quant_config = dict(layer_quant_config)

        num_moe_layers = pretrained_config.num_hidden_layers
        if (spec_config is not None
                and spec_config.spec_dec_mode.is_mtp_one_model()):
            num_moe_layers += spec_config.num_nextn_predict_layers

        for layer_idx in range(num_moe_layers):
            layer_quant_config[
                f"model.layers.{layer_idx}.mlp.experts"] = experts_quant_config

        logger.info(
            "Detected DeepSeek-V4 routed MoE %s checkpoint layout; using "
            "%s for routed experts.", layout.upper(),
            experts_quant_config.quant_algo)
        return layer_quant_config

    @staticmethod
    def _set_minimax_m3_layer_quant_config(pretrained_config,
                                           layer_quant_config):
        """Normalize the Minimax M3 MIXED_PRECISION per-layer quant config.

        Two fix-ups are applied:

        1. Strip the ``language_model.`` prefix from every per-layer key.
           The M3 VL checkpoint stores keys like
           ``language_model.model.layers.0.self_attn.o_proj -> MXFP8`` in
           ``hf_quant_config.json``, but the TRT-LLM module tree names the text
           decoder ``model.layers.0.self_attn.o_proj`` (no ``language_model.``
           prefix -- the loader strips it). ``apply_layerwise_quant_config``
           matches *standalone* Linears (e.g. ``o_proj``, ``down_proj``) with an
           **exact** ``name == key`` comparison, so the prefixed keys never match
           and those layers silently fall back to the global ``MIXED_PRECISION``
           config -> loaded unquantized (MXFP8 ``weight_scale`` dropped) -> the
           attention/MLP output magnitude explodes. Stripping the prefix makes
           the exact match succeed. (Fused qkv/gate_up Linears and the Attention
           wrapper use substring matches and happened to work regardless.)

        2. Inject a single coarse ``model.layers.N.block_sparse_moe.experts``
           entry per MoE layer so ``MiniMaxM3MoE._get_experts_quant_config`` can
           select the NVFP4 backend for the routed experts (the fine-grained
           per-linear NVFP4 expert keys can't be used directly).

        Does nothing when there is no per-layer config (e.g. the uniform MXFP8
        or a BF16 checkpoint).
        """
        from tensorrt_llm.models.modeling_utils import QuantAlgo
        if layer_quant_config is None:
            return layer_quant_config

        # (1) Strip the ``language_model.`` prefix so exact-match per-layer
        # quant assignment works for standalone base Linears.
        _LM_PREFIX = "language_model."
        layer_quant_config = {
            (k[len(_LM_PREFIX):] if k.startswith(_LM_PREFIX) else k): v
            for k, v in layer_quant_config.items()
        }

        # (2) Inject coarse NVFP4 expert entries (only when routed experts are
        # NVFP4).
        has_nvfp4_experts = any(
            "block_sparse_moe.experts" in k and isinstance(v, QuantConfig)
            and v.quant_algo == QuantAlgo.NVFP4
            for k, v in layer_quant_config.items())
        if not has_nvfp4_experts:
            return layer_quant_config

        experts_quant_config = QuantConfig()
        experts_quant_config.quant_algo = QuantAlgo.NVFP4
        # TODO: remove the hardcoded group_size and read it from the per-linear
        # NVFP4 expert entries in hf_quant_config.json instead. 16 is correct
        # for standard NVFP4 today, but this is a latent bug if a checkpoint
        # ever ships a different group size.
        experts_quant_config.group_size = 16

        text_config = getattr(pretrained_config, "text_config",
                              pretrained_config)
        if isinstance(text_config, dict):
            moe_layer_freq = text_config.get("moe_layer_freq", [])
        else:
            moe_layer_freq = getattr(text_config, "moe_layer_freq", [])

        for layer_idx, freq in enumerate(moe_layer_freq):
            if int(freq) != 0:
                layer_quant_config[
                    f"model.layers.{layer_idx}.block_sparse_moe.experts"] = experts_quant_config

        logger.info(
            "Detected Minimax M3 NVFP4 routed MoE checkpoint; using NVFP4 "
            "for routed experts.")
        return layer_quant_config

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

        def update_sparse_attention_indexer_config(pretrained_config, kwargs):
            sparse_attention_config = kwargs.get('sparse_attention_config')
            if sparse_attention_config:
                index_n_heads = sparse_attention_config.index_n_heads or pretrained_config.index_n_heads
                index_head_dim = sparse_attention_config.index_head_dim or pretrained_config.index_head_dim
                # index_topk needs an explicit-set check rather than `or`: the
                # DeepSeekV4SparseAttentionConfig default (512) is truthy, so a
                # plain `or` shadows the checkpoint's index_topk (e.g. Pro's
                # 1024) whenever the user did not set it. Mirror the window_size
                # handling below and consult model_fields_set. (index_n_heads /
                # index_head_dim stay on `or` since their defaults are None.)
                if 'index_topk' in sparse_attention_config.model_fields_set:
                    index_topk = sparse_attention_config.index_topk
                else:
                    index_topk = pretrained_config.index_topk
                indexer_max_chunk_size = sparse_attention_config.indexer_max_chunk_size
                skip_indexer_for_short_seqs = sparse_attention_config.skip_indexer_for_short_seqs
                # Pass-through DSA tuning flags so user-set values survive the
                # V4 sparse_attention_config rebuild below. The V3.2 path
                # already threads these explicitly (lines 723-727); without
                # this block the V4 rebuild silently drops any user override
                # back to subclass defaults (e.g., enable_heuristic_topk=False
                # even when the user set it to True in --extra_llm_api_options).
                use_cute_dsl_topk = sparse_attention_config.use_cute_dsl_topk
                use_cute_dsl_paged_mqa_logits = sparse_attention_config.use_cute_dsl_paged_mqa_logits
                q_split_threshold = sparse_attention_config.q_split_threshold
                indexer_rope_interleave = sparse_attention_config.indexer_rope_interleave
                enable_heuristic_topk = sparse_attention_config.enable_heuristic_topk
                indexer_k_dtype = sparse_attention_config.indexer_k_dtype
            else:
                index_n_heads = pretrained_config.index_n_heads
                index_head_dim = pretrained_config.index_head_dim
                index_topk = pretrained_config.index_topk
                indexer_max_chunk_size = None
                skip_indexer_for_short_seqs = True
                # Defaults match DeepSeekV4SparseAttentionConfig field defaults.
                use_cute_dsl_topk = False
                use_cute_dsl_paged_mqa_logits = False
                q_split_threshold = 8192
                indexer_rope_interleave = False
                enable_heuristic_topk = False
                default_sparse_attention_config = DeepSeekV4SparseAttentionConfig(
                )
                indexer_k_dtype = default_sparse_attention_config.indexer_k_dtype
            indexer_config = {}
            indexer_config['index_n_heads'] = index_n_heads
            indexer_config['index_head_dim'] = index_head_dim
            indexer_config['index_topk'] = index_topk
            indexer_config['indexer_max_chunk_size'] = indexer_max_chunk_size
            indexer_config[
                'skip_indexer_for_short_seqs'] = skip_indexer_for_short_seqs
            indexer_config['use_cute_dsl_topk'] = use_cute_dsl_topk
            indexer_config[
                'use_cute_dsl_paged_mqa_logits'] = use_cute_dsl_paged_mqa_logits
            indexer_config['q_split_threshold'] = q_split_threshold
            indexer_config['indexer_rope_interleave'] = indexer_rope_interleave
            indexer_config['enable_heuristic_topk'] = enable_heuristic_topk
            indexer_config['indexer_k_dtype'] = indexer_k_dtype
            return indexer_config

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
                if pretrained_config.architectures[0] in [
                        "DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"
                ]:
                    sparse_attention_config = kwargs.get(
                        'sparse_attention_config')
                    indexer_rope_interleave = getattr(
                        pretrained_config, 'indexer_rope_interleave', False)
                    if sparse_attention_config:
                        index_n_heads = sparse_attention_config.index_n_heads or pretrained_config.index_n_heads
                        index_head_dim = sparse_attention_config.index_head_dim or pretrained_config.index_head_dim
                        # Explicit-set check (see V4 path above): only honor a
                        # user-provided index_topk; otherwise take the
                        # checkpoint value rather than a truthy subclass default.
                        if 'index_topk' in sparse_attention_config.model_fields_set:
                            index_topk = sparse_attention_config.index_topk
                        else:
                            index_topk = pretrained_config.index_topk
                        indexer_max_chunk_size = sparse_attention_config.indexer_max_chunk_size
                        skip_indexer_for_short_seqs = sparse_attention_config.skip_indexer_for_short_seqs
                        use_cute_dsl_topk = sparse_attention_config.use_cute_dsl_topk
                        use_cute_dsl_paged_mqa_logits = sparse_attention_config.use_cute_dsl_paged_mqa_logits
                        q_split_threshold = sparse_attention_config.q_split_threshold
                        enable_heuristic_topk = sparse_attention_config.enable_heuristic_topk
                        indexer_k_dtype = sparse_attention_config.indexer_k_dtype
                    else:
                        index_n_heads = pretrained_config.index_n_heads
                        index_head_dim = pretrained_config.index_head_dim
                        index_topk = pretrained_config.index_topk
                        indexer_max_chunk_size = None
                        skip_indexer_for_short_seqs = True
                        use_cute_dsl_topk = False
                        use_cute_dsl_paged_mqa_logits = False
                        q_split_threshold = 8192
                        enable_heuristic_topk = False
                        indexer_k_dtype = "fp8"
                    kwargs[
                        'sparse_attention_config'] = DeepSeekSparseAttentionConfig(
                            index_n_heads=index_n_heads,
                            index_head_dim=index_head_dim,
                            index_topk=index_topk,
                            indexer_max_chunk_size=indexer_max_chunk_size,
                            skip_indexer_for_short_seqs=
                            skip_indexer_for_short_seqs,
                            use_cute_dsl_topk=use_cute_dsl_topk,
                            use_cute_dsl_paged_mqa_logits=
                            use_cute_dsl_paged_mqa_logits,
                            q_split_threshold=q_split_threshold,
                            indexer_rope_interleave=indexer_rope_interleave,
                            enable_heuristic_topk=enable_heuristic_topk,
                            indexer_k_dtype=indexer_k_dtype)
                elif pretrained_config.architectures[
                        0] == "DeepseekV4ForCausalLM":
                    if cls._is_deepseek_v4_base_checkpoint(checkpoint_dir):
                        logger.warning(
                            "Support for DeepSeek-V4 Base checkpoints is "
                            "experimental. For better supported behavior, use "
                            "a DeepSeek-V4 Instruct checkpoint.")
                    indexer_config = update_sparse_attention_indexer_config(
                        pretrained_config, kwargs)
                    checkpoint_compress_ratios = getattr(
                        pretrained_config, 'compress_ratios', None)
                    num_base_layers = pretrained_config.num_hidden_layers
                    spec_config = kwargs.get('spec_config', None)
                    if (spec_config is not None
                            and getattr(spec_config, 'num_nextn_predict_layers',
                                        None) is None):
                        spec_config.num_nextn_predict_layers = getattr(
                            pretrained_config, 'num_nextn_predict_layers', 1)
                    mtp_enabled = (spec_config is not None and
                                   spec_config.spec_dec_mode.is_mtp_one_model())
                    sparse_attention_config = kwargs.get(
                        'sparse_attention_config')
                    checkpoint_window_size = getattr(pretrained_config,
                                                     'window_size', None)
                    if checkpoint_window_size is None:
                        checkpoint_window_size = getattr(
                            pretrained_config, 'sliding_window', None)
                    if sparse_attention_config:
                        compress_ratios = sparse_attention_config.compress_ratios
                        window_size = sparse_attention_config.window_size
                        if 'window_size' not in sparse_attention_config.model_fields_set:
                            window_size = checkpoint_window_size
                    else:
                        compress_ratios = checkpoint_compress_ratios
                        window_size = checkpoint_window_size

                    if (checkpoint_compress_ratios is not None
                            and (compress_ratios is None
                                 or len(checkpoint_compress_ratios)
                                 > len(compress_ratios))):
                        compress_ratios = checkpoint_compress_ratios

                    if window_size is None:
                        window_size = checkpoint_window_size
                    if window_size is None:
                        window_size = pretrained_config.sliding_window

                    # Normalize checkpoint-facing ratio 0 (SWA-only/uncompressed)
                    # to 1 internally so cache allocation math works. The
                    # external config keeps the original semantics.
                    compress_ratios = [
                        ratio if ratio > 0 else 1 for ratio in compress_ratios
                    ]

                    # Only synthesize ratios for extra MTP layers. The base
                    # model ratios must come from the checkpoint or an
                    # explicit user override; padding a short default list for
                    # non-MTP changes sparse attention semantics.
                    if mtp_enabled:
                        mtp_num_layers = spec_config.num_nextn_predict_layers
                        total_layers = num_base_layers + mtp_num_layers
                        if len(compress_ratios) < total_layers:
                            compress_ratios = list(compress_ratios) + [1] * (
                                total_layers - len(compress_ratios))

                    indexer_k_dtype = indexer_config.pop('indexer_k_dtype')
                    kwargs[
                        'sparse_attention_config'] = DeepSeekV4SparseAttentionConfig(
                            compress_ratios=compress_ratios,
                            window_size=window_size,
                            indexer_k_dtype=indexer_k_dtype,
                            **indexer_config)
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
        dtype = getattr(pretrained_config, 'dtype', None)
        # For composite VLM configs the dtype lives inside ``text_config``
        # because the top-level config has no ``dtype`` field.
        if dtype is None:
            text_config = getattr(pretrained_config, 'text_config', None)
            if text_config is not None:
                dtype = getattr(text_config, 'dtype', None)
        pretrained_config.torch_dtype = dtype

        # Prior to transformers 5, composite configs (e.g. Qwen2_5_VLConfig) delegated attribute
        # lookups to their text sub-config, so accesses like `config.vocab_size` /
        # `config.hidden_size` resolved transparently.
        # 5.x removed that delegation, so eagerly mirror the text sub-config onto  the top-level
        # config to keep downstream consumers working.
        _mirror_text_subconfig_attrs(pretrained_config)

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
        require_deepseek_v4_routed_moe_layout = False
        requested_moe_backend = kwargs.get('moe_backend', 'AUTO')
        architecture = pretrained_config.architectures[
            0] if pretrained_config.architectures else ""
        # Use an architecture-only backend hint for quant config parsing. Some
        # quant formats choose the quant_algo from the backend name, so the final
        # quant-aware AUTO resolution happens after quant_config is loaded.
        moe_backend_hint = cls.resolve_moe_backend(requested_moe_backend,
                                                   architecture)

        # quantized ckpt in modelopt format
        if quant_config_file := cached_file(checkpoint_dir,
                                            'hf_quant_config.json'):
            with open(quant_config_file) as f:
                normalized = read_modelopt_quant_config(json.load(f))
            # The file is authoritative; warn if the inline copy disagrees.
            # Done before _build_modelopt_quant_config since the builder may
            # mutate ``normalized`` via ``.update`` from quant_cfg.json.
            warn_if_inline_diverges(
                normalized,
                getattr(pretrained_config, "quantization_config", None),
                source_file="hf_quant_config.json",
            )
            if architecture in _DEEPSEEK_V4_ARCHITECTURES:
                require_deepseek_v4_routed_moe_layout = (
                    cls._has_deepseek_v4_layer_only_modelopt_quant_config(
                        quant_config_file))
            quant_config, layer_quant_config = cls._build_modelopt_quant_config(
                normalized, checkpoint_dir, moe_backend_hint)
            hf_quant_config = getattr(pretrained_config, "quantization_config",
                                      None)
            if quant_config.quant_algo is None and hf_quant_config is not None:
                hf_quant_config, hf_layer_quant_config = cls.load_hf_quant_config(
                    hf_quant_config,
                    moe_backend_hint,
                    checkpoint_dir=checkpoint_dir)
                if hf_quant_config.quant_algo is not None:
                    logger.info(
                        "Using quantization_config from config.json as global "
                        "quantization because hf_quant_config.json does not set "
                        "a global quant_algo.")
                    quant_config = hf_quant_config
                    if hf_layer_quant_config is not None:
                        if layer_quant_config is None:
                            layer_quant_config = hf_layer_quant_config
                        else:
                            layer_quant_config = {
                                **hf_layer_quant_config,
                                **layer_quant_config,
                            }
        # quantized ckpt in other formats
        elif getattr(pretrained_config, "quantization_config",
                     None) is not None:
            hf_quant_config = pretrained_config.quantization_config
            quant_config, layer_quant_config = cls.load_hf_quant_config(
                hf_quant_config,
                moe_backend_hint,
                checkpoint_dir=checkpoint_dir)
        elif quant_config_file := cached_file(checkpoint_dir, 'dtypes.json'):
            quant_config, layer_quant_config = cls.load_quant_config_from_dtypes_json(
                quant_config_file, moe_backend_hint)

        kwargs['moe_backend'] = cls.resolve_moe_backend(
            requested_moe_backend,
            architecture,
            quant_config=quant_config,
        )

        if architecture in _DEEPSEEK_V4_ARCHITECTURES:
            layer_quant_config = cls._set_deepseek_v4_routed_moe_quant_config(
                pretrained_config,
                checkpoint_dir,
                kwargs['moe_backend'],
                layer_quant_config,
                kwargs.get('spec_config', None),
                require_layout=require_deepseek_v4_routed_moe_layout)

        if architecture in _MINIMAX_M3_ARCHITECTURES:
            layer_quant_config = cls._set_minimax_m3_layer_quant_config(
                pretrained_config, layer_quant_config)

        model_config = cls(pretrained_config=pretrained_config,
                           quant_config=quant_config,
                           quant_config_dict=layer_quant_config,
                           **kwargs)
        model_config._frozen = True
        return model_config

    def get_bindings_model_config(
        self,
        is_disagg: bool = False,
        tokens_per_block: Optional[int] = None,
        kv_cache_config: Optional[KvCacheConfig] = None,
        spec_config: Optional['SpeculativeConfig'] = None,
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

        def ceil_div(a, b):
            return (a + b - 1) // b

        num_heads = ceil_div(self.pretrained_config.num_attention_heads,
                             attn_tp_size * attn_cp_size)

        hidden_size = ceil_div(self.pretrained_config.hidden_size, attn_tp_size)
        num_layers = self.pretrained_config.num_hidden_layers
        num_attention_layers = self.get_num_attention_layers()
        if (self.spec_config is not None
                and self.spec_config.spec_dec_mode.is_mtp_one_model()):
            assert self.spec_config.num_nextn_predict_layers is not None, (
                "num_nextn_predict_layers must be set from model config before building ModelConfig. "
                "Ensure update_spec_config_from_model_config() has been called."
            )
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
                ceil_div(kv_heads, attn_tp_size * attn_cp_size)
                for kv_heads in num_key_value_heads
            ]
            model_config_cpp.num_kv_heads_per_layer = num_kv_heads_per_layer
        else:
            num_kv_heads = ceil_div(num_key_value_heads,
                                    attn_tp_size * attn_cp_size)
            model_config_cpp.set_num_kv_heads(num_kv_heads)

        # For hybrid models (e.g., Nemotron-H with Mamba + Attention), LoRA can be applied
        # to non-attention layers (e.g., Mamba in_proj/out_proj). Set num_lora_layers to
        # total layers so the C++ LoRA validation accepts all layer indices.
        if is_nemotron_hybrid(self.pretrained_config):
            model_config_cpp.set_num_lora_layers(num_layers)

        mlp_hidden_size = None
        if self.pretrained_config.intermediate_size is not None:
            mlp_hidden_size = ceil_div(self.pretrained_config.intermediate_size,
                                       self.mapping.tp_size)
        else:
            # TODO: once tensorrt_llm._torch.AutoConfig is implemented, the following logic
            # should be moved to tensorrt_llm._torch.AutoConfig of the relevant modeling_xxx file
            if hasattr(self.pretrained_config, "architectures"
                       ) and self.pretrained_config.architectures is not None:
                architectures = self.pretrained_config.architectures
                if len(architectures
                       ) == 1 and architectures[0] == "DeciLMForCausalLM":
                    mlp_hidden_size = ceil_div(self._infer_nemotron_ffn_mult(),
                                               self.mapping.tp_size)
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

    def get_num_attention_layers(self) -> int:
        """Number of full-attention layers in the model.

        Pure model property: independent of which KV cache manager will run.
        For non-hybrid models this equals num_hidden_layers. For hybrid Mamba
        models (Nemotron-hybrid, Qwen3-hybrid) it returns only the attention
        count derived from the layer pattern; mamba layers are reported by
        ``get_num_mamba_layers``.
        """
        cfg = self.pretrained_config
        if is_nemotron_hybrid(cfg):
            return cfg.hybrid_override_pattern.count("*")
        if is_qwen3_hybrid(cfg):
            return get_qwen3_hybrid_num_attention_layers(cfg)
        return cfg.num_hidden_layers

    def get_num_mamba_layers(self) -> int:
        """Number of Mamba / linear-attention layers (0 for non-hybrid)."""
        cfg = self.pretrained_config
        if is_nemotron_hybrid(cfg):
            return cfg.hybrid_override_pattern.count("M")
        if is_qwen3_hybrid(cfg):
            return cfg.num_hidden_layers - get_qwen3_hybrid_num_attention_layers(
                cfg)
        return 0


def _mirror_text_subconfig_attrs(
        pretrained_config: transformers.PretrainedConfig) -> None:
    """Mirror text sub-config attributes onto the parent config.

    Composite configs (e.g. Qwen2_5_VLConfig) keep text-side fields like `vocab_size`,
    `hidden_size`, `num_attention_heads`, etc. inside a `text_config` sub-config.
    Prior to transformers 5, the parent config delegated attribute lookups there automatically;
    that delegation was removed in 5.x.

    Copying the sub-config's attributes onto the parent keeps downstream code (which accesses these
    on the top-level config) working without having to learn about the composite layout.
    """
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is None or not isinstance(text_config,
                                             transformers.PretrainedConfig):
        return
    for key, value in vars(text_config).items():
        if key.startswith("_"):
            continue
        try:
            getattr(pretrained_config, key)
        except AttributeError:
            setattr(pretrained_config, key, value)
