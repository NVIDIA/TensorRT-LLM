import torch

from ..models.modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_utils import register_auto_model
from ..model_config import ModelConfig

from torch import nn
from typing import Dict, Optional, List

from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import (
    MistralLarge3WeightMapper,
)
from tensorrt_llm._torch.modules.fused_moe import RenormalizeNaiveMoeRoutingMethod


class Mistral3Gate(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size),
                                               dtype=dtype),
                                   requires_grad=False)
        self.top_k = top_k
        self.dtype = dtype
        self.routing_method = RenormalizeNaiveMoeRoutingMethod(top_k=self.top_k)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = torch.ops.trtllm.cublas_mm(hidden_states,
                                                          self.weight.t(),
                                                          bias=None,
                                                          out_dtype=self.dtype)
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

@register_auto_model("MistralLarge3ForCausalLM")
class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def load_weights(self, weights: Dict, *args, **kwargs):
        super().load_weights(llm_weights)
