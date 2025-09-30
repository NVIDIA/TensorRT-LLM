from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE


@register_mapper("HF", "Qwen2MoeForCausalLM")
class Qwen2MoeHfWeightMapper(HfWeightMapper):

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoE)

    def handle_special_instance_module(self, module: nn.Module,
                                       module_name: str,
                                       module_weights: dict) -> None:
        if isinstance(module, MoE):
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                new_weight_name = weight_name.replace(
                    "gate_proj", "w1").replace("up_proj",
                                               "w3").replace("down_proj", "w2")
                updated_module_weights[new_weight_name] = weight_value
            del module_weights
            module.load_weights(weights=[updated_module_weights])
