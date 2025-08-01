from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "MixtralForCausalLM")
class MixtralHfWeightMapper(BaseWeightMapper):

    def map_weights(self) -> None:
        self.mapping.update({
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        })

    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        module_weights = []

        for new_name in self.mapping[module_name]:
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)
            module_weights.append(fw)

        return module_weights
