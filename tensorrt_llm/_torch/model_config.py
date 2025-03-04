import json
import os
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, TypeVar

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

    attn_backend: str = 'TRTLLM'

    is_generation: bool = True

    @property
    def fuse_pos_embd(self):
        if self.attn_backend == 'TRTLLM':
            return True
        elif self.attn_backend == 'FLASHINFER':
            return False
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
            "Qwen2ForProcessRewardModel", "BertForSequenceClassification"
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

        quant_config = QuantConfig()
        quant_config_file = os.path.join(checkpoint_dir, 'hf_quant_config.json')

        # quantized ckpt in ModelOpt format
        if os.path.exists(quant_config_file):
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
        # quantized ckpt in other formats
        elif hasattr(pretrained_config, "quantization_config"):
            hf_quant_config = pretrained_config.quantization_config
            # DeepSeek V3 FP8 ckpt
            if hf_quant_config.get(
                    "quant_method") == "fp8" and hf_quant_config.get(
                        "weight_block_size", []):
                quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                quant_config.exclude_modules = ["*eh_proj"]

        is_generation = cls.is_generation_model(pretrained_config.architectures)

        return cls(pretrained_config=pretrained_config,
                   quant_config=quant_config,
                   is_generation=is_generation,
                   **kwargs)
