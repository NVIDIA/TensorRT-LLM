from typing import Union

from torch import nn

from tensorrt_llm._torch.model_config import TConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM


@register_mapper("HF", "DeciLMForCausalLM")
class NemotronNASHFWeightMapper(HfWeightMapper):

    def init_model_and_config(self, model: Union[nn.Module,
                                                 DecoderModelForCausalLM],
                              config: TConfig):
        super().init_model_and_config(model, config)
        self._head_dim = getattr(
            self.config.pretrained_config, "head_dim",
            self.config.pretrained_config.hidden_size //
            self.config.pretrained_config.num_attention_heads)

    def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
                              weights: dict):
        if new_name in ['k_proj', 'v_proj']:
            assert weights["weight"].shape[0] % self._head_dim == 0
            num_kv_heads = weights["weight"].shape[0] // self._head_dim
            processed_weights = {
                k:
                self._duplicate_kv(weight=v[:],
                                   num_kv_heads=num_kv_heads,
                                   tensor_parallel_size=self._tp_size)
                if k in ["weight", "bias"] else v
                for k, v in weights.items()
            }
            return processed_weights

        return weights
