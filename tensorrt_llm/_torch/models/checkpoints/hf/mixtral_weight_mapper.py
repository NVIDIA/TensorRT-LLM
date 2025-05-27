from typing import Union

from torch import nn

from tensorrt_llm._torch.model_config import TConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import (
    BaseWeightMapper, guard_all_methods)
from tensorrt_llm._torch.models.modeling_utils import (DecoderModelForCausalLM,
                                                       register_mapper)


@register_mapper("HF", "MixtralForCausalLM")
@guard_all_methods
class MixtralHfWeightMapper(BaseWeightMapper):

    def init(self, model: Union[nn.Module, DecoderModelForCausalLM],
             config: TConfig):
        super().init(model, config)

    def map_weights(self) -> None:
        self._mapping.update({
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        })

    def apply_callbacks(self, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        module_weights = []

        for new_name in self._mapping[module_name]:
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)
            module_weights.append(fw)

        return module_weights
