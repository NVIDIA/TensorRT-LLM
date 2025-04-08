import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar

import torch
import transformers

from tensorrt_llm import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)


@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    pretrained_config: Optional[TConfig] = None
    mapping: Mapping = field(default_factory=Mapping)
    quant_config: QuantConfig = field(default_factory=QuantConfig)
    # TODO(qijun): support per linear layer quantization
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    skip_create_weights: bool = False
    is_generation: bool = True
    max_num_tokens: int = 8192
    moe_max_num_tokens: Optional[int] = None

    attn_backend: str = 'TRTLLM'

    def __post_init__(self):
        if self.pretrained_config and hasattr(self.pretrained_config,
                                              "architectures"):
            self.is_generation = self.is_generation_model(
                self.pretrained_config.architectures)

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
            if hasattr(self.pretrained_config, "kv_lora_rank") and hasattr(
                    self.pretrained_config, "qk_rope_head_dim"):
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
    def is_generation_model(model_architectures: Optional[List[str]]) -> bool:
        if model_architectures is None:
            logger.warning(
                "Model architectures is None, default to is_generation_model=True"
            )
            return True
        return model_architectures[0] not in [
            "BertForSequenceClassification", "Qwen2ForProcessRewardModel",
            "Qwen2ForRewardModel"
        ]
        # TODO: should be 'not model_type == ModelType.ENCODER_ONLY'
        # once ModelType is used in pytorch flow.

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str,
                        trust_remote_code=False,
                        **kwargs):
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
        # quantized ckpt in modelopt format
        quant_config_file = model_dir / 'hf_quant_config.json'
        if quant_config_file.exists():
            with open(quant_config_file) as f:
                quant_config_dict = json.load(f)

            json_quant_configs = quant_config_dict['quantization']

            def _load_json_quant_config(key: str):
                if key in json_quant_configs:
                    return json_quant_configs[key]
                return None

            quant_config.quant_algo = _load_json_quant_config('quant_algo')
            quant_config.kv_cache_quant_algo = _load_json_quant_config(
                'kv_cache_quant_algo')
            quant_config.group_size = _load_json_quant_config('group_size')
            quant_config.exclude_modules = _load_json_quant_config(
                'exclude_modules')

            if quant_config.quant_algo == QuantAlgo.MIXED_PRECISION:
                mixed_quant_config_file = model_dir / 'quant_cfg.json'
                with open(mixed_quant_config_file) as fm:
                    mixed_quant_config = json.load(fm)
                    mixed_quant_config = mixed_quant_config['quantized_layers']
                    for k in mixed_quant_config:
                        config = QuantConfig()
                        config.quant_algo = mixed_quant_config[k]['quant_algo']
                        mixed_quant_config[k] = config
                layer_quant_config = mixed_quant_config
        # quantized ckpt in other formats
        elif hasattr(pretrained_config, "quantization_config"):
            hf_quant_config = pretrained_config.quantization_config
            # DeepSeek V3 FP8 ckpt
            if hf_quant_config.get(
                    "quant_method") == "fp8" and hf_quant_config.get(
                        "weight_block_size", []):
                quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                quant_config.exclude_modules = ["*eh_proj"]

        return cls(pretrained_config=pretrained_config,
                   quant_config=quant_config,
                   quant_config_dict=layer_quant_config,
                   **kwargs)
