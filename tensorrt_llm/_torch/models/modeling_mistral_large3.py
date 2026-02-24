from typing import Dict, List

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import MistralLarge3WeightMapper
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM
from tensorrt_llm._torch.models.modeling_utils import register_auto_model
from tensorrt_llm._torch.modules.fused_moe import RenormalizeNaiveMoeRoutingMethod
from tensorrt_llm.quantization.mode import QuantAlgo


class Mistral3Gate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=dtype), requires_grad=False
        )
        self.top_k = top_k
        self.dtype = dtype
        self.routing_method = RenormalizeNaiveMoeRoutingMethod(top_k=self.top_k)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = torch.ops.trtllm.cublas_mm(
            hidden_states, self.weight.t(), bias=None, out_dtype=self.dtype
        )
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])


@register_auto_model("MistralLarge3ForCausalLM")
class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.weight_mapper = MistralLarge3WeightMapper()

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def load_weights(self, weights: Dict):
        assert self.model_config is not None, "self.model_config is required"
        params_map = self.weight_mapper.mistral_llm_mapping.copy()
        quantization_weights_map: Dict[str, str] = {}
        if self.model_config.quant_config.quant_algo == QuantAlgo.NVFP4:
            quantization_weights_map = {
                "weight_packed": "weight",
                "input_global_scale": "input_scale",
                "weight_global_scale": "weight_scale_2",
            }
        elif self.model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            quantization_weights_map = {
                "weight_scale": "weight_scale_inv",
            }
        if quantization_weights_map:
            params_map.update(quantization_weights_map)
        weights = self.weight_mapper.rename_by_params_map(weights=weights, params_map=params_map)

        super().load_weights(weights)
