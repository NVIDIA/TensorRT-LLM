import json
import os
from dataclasses import dataclass, field
from typing import Dict, Generic, Optional, TypeVar

import transformers

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)


@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    pretrained_config: Optional[TConfig] = None
    mapping: Mapping = field(default_factory=Mapping)
    quant_config: QuantConfig = field(default_factory=QuantConfig)
    # TODO(qijun): support per linear layer quantization
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None

    attn_backend: str = 'TRTLLM'

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

        return cls(pretrained_config=pretrained_config,
                   quant_config=quant_config,
                   **kwargs)
