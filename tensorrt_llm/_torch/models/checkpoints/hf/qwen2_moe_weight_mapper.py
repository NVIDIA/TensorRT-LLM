import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE


def _unfuse_moe_expert_weights(weights: dict) -> dict:
    """Unfuse HF transformers 5.x fused MoE expert weights to per-expert format.

    Transforms:
      gate_up_proj [num_experts, 2*intermediate, hidden]
        -> {i}.gate_proj.weight, {i}.up_proj.weight
      down_proj [num_experts, hidden, intermediate]
        -> {i}.down_proj.weight
    """
    has_fused = any(k in weights
                    for k in ("gate_up_proj", "down_proj",
                              "gate_up_proj_scale_inv", "down_proj_scale_inv"))
    if not has_fused:
        return weights

    updated = {}
    for key, value in weights.items():
        if not isinstance(value, torch.Tensor):
            updated[key] = value
            continue

        # Fused gate_up_proj [num_experts, 2*intermediate, hidden]
        if key == "gate_up_proj" and value.dim() == 3:
            num_experts = value.shape[0]
            half = value.shape[1] // 2
            for i in range(num_experts):
                updated[f"{i}.gate_proj.weight"] = value[i, :half, :]
                updated[f"{i}.up_proj.weight"] = value[i, half:, :]
        # Fused gate_up_proj FP8 scales [num_experts, 2*intermediate_blocks, hidden_blocks]
        elif key == "gate_up_proj_scale_inv" and value.dim() == 3:
            num_experts = value.shape[0]
            half = value.shape[1] // 2
            for i in range(num_experts):
                updated[f"{i}.gate_proj.weight_scale_inv"] = value[i, :half, :]
                updated[f"{i}.up_proj.weight_scale_inv"] = value[i, half:, :]
        # Fused down_proj [num_experts, hidden, intermediate]
        elif key == "down_proj" and value.dim() == 3:
            num_experts = value.shape[0]
            for i in range(num_experts):
                updated[f"{i}.down_proj.weight"] = value[i]
        # Fused down_proj FP8 scales [num_experts, hidden_blocks, intermediate_blocks]
        elif key == "down_proj_scale_inv" and value.dim() == 3:
            num_experts = value.shape[0]
            for i in range(num_experts):
                updated[f"{i}.down_proj.weight_scale_inv"] = value[i]
        else:
            updated[key] = value
    return updated


@register_mapper("HF", "Qwen2MoeForCausalLM")
class Qwen2MoeHfWeightMapper(HfWeightMapper):

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoE)

    def handle_special_instance_module(
            self,
            module: nn.Module,
            module_name: str,
            module_weights: dict,
            allow_partial_loading: bool = False) -> None:
        if isinstance(module, MoE):
            # Transformers 5.x uses fused expert weights:
            #   gate_up_proj [num_experts, 2*intermediate, hidden]
            #   down_proj [num_experts, hidden, intermediate]
            # Unfuse them to per-expert format before renaming.
            module_weights = _unfuse_moe_expert_weights(module_weights)
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                new_weight_name = weight_name.replace(
                    "gate_proj", "w1").replace("up_proj",
                                               "w3").replace("down_proj", "w2")
                updated_module_weights[new_weight_name] = weight_value
            del module_weights
            module.load_weights(weights=[updated_module_weights],
                                allow_partial_loading=allow_partial_loading)
