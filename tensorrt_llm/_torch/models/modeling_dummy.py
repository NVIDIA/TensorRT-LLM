from typing import Dict, List, Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.models.modeling_utils import (ModelConfig,
                                                       register_auto_model)


class DummyConfig(PretrainedConfig):

    def __init__(self):
        self.architectures: list[str] = ["DummyModel"]
        self.torch_dtype: torch.dtype = torch.float16
        self.num_key_value_heads: int = 16
        self.num_attention_heads: int = 16
        self.hidden_size: int = 256
        self.vocab_size: int = 1000
        self.num_hidden_layers: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@register_auto_model("DummyModel")
class DummyModel(torch.nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config
        self.recorded_position_ids = None

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        self.recorded_position_ids = kwargs["position_ids"]
        batch_size = input_ids.size(0)
        return {
            "logits": torch.randn((batch_size, 10), device='cuda'),
            "context_output": torch.randn((batch_size, 10), device='cuda'),
            "generation_output": torch.randn((batch_size, 10), device='cuda')
        }

    def load_weights(self,
                     weights: Dict,
                     weight_mapper: Optional["BaseWeightMapper"] = None,
                     skip_modules: List[str] = []):
        pass
