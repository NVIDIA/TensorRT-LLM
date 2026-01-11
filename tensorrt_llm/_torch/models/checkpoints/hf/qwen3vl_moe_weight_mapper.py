from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_moe_weight_mapper import Qwen3MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE


@register_mapper("HF", "Qwen3VLMoeForConditionalGeneration")
class Qwen3VLMoeHfWeightMapper(Qwen3MoeHfWeightMapper):
    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if isinstance(module, MoE):
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                new_weight_name = weight_name.replace("scale_inv", "weight_scale")
                updated_module_weights[new_weight_name] = weight_value
            module.load_weights(
                weights=[updated_module_weights], allow_partial_loading=allow_partial_loading
            )
