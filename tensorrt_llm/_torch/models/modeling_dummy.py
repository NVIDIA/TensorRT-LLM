from typing import Dict, List, Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
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

    def forward(self,
                *args,
                input_ids: torch.Tensor,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False,
                **kwargs) -> torch.Tensor:
        num_batch_tokens = input_ids.size(0)

        vocab_size = self.config.vocab_size
        hidden_size = self.config.hidden_size

        last_tokens = torch.cumsum(
            attn_metadata.seq_lens_cuda,
            dim=0,
            dtype=torch.long,
        ) - 1

        # Logits: fixed values for testing
        logits = torch.ones((num_batch_tokens, vocab_size), device='cuda') * 0.1

        # Logits shape depends on return_context_logits flag
        if not return_context_logits:
            # For context logits, return logits for all positions
            logits = logits[last_tokens]

        # Context output: fixed values for testing, one output per input token
        context_output = torch.ones(
            (num_batch_tokens, hidden_size), device='cuda') * 0.2

        # Generation output: fixed values for testing, one output per sequence
        generation_output = torch.ones(
            (num_batch_tokens, hidden_size), device='cuda') * 0.3
        generation_output = generation_output[last_tokens]

        return {
            "logits": logits,
            "context_output": context_output,
            "generation_output": generation_output
        }

    def load_weights(self,
                     weights: Dict,
                     weight_mapper: Optional[BaseWeightMapper] = None,
                     skip_modules: List[str] = []):
        pass
