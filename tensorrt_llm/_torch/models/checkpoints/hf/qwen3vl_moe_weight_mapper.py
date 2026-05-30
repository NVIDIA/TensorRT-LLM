from torch import nn
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)

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
            # Qwen3VL MoE uses MoEWeightLoadingMode.FUSED_GATE_UP_PROJ, whose
            # loader expects gate_up_proj in [E, H, 2*I] and down_proj in
            # [E, I, H] format, but HF stores them transposed:
            #   gate_up_proj: [E, 2*I, H]  ->  transpose to [E, H, 2*I]
            #   down_proj:    [E, H,  I]   ->  transpose to [E, I,  H]
            # Also rename FP8 scale_inv -> weight_scale.
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                if weight_name == "gate_up_proj" and weight_value.ndim == 3:
                    if (
                        weight_value.shape[-2] == 2 * module.intermediate_size
                        and weight_value.shape[-1] == module.hidden_size
                    ):
                        weight_value = weight_value.transpose(-1, -2).contiguous()
                elif weight_name == "down_proj" and weight_value.ndim == 3:
                    if (
                        weight_value.shape[-2] == module.hidden_size
                        and weight_value.shape[-1] == module.intermediate_size
                    ):
                        weight_value = weight_value.transpose(-1, -2).contiguous()
                new_weight_name = weight_name.replace("scale_inv", "weight_scale")
                updated_module_weights[new_weight_name] = weight_value
            module.load_weights(
                weights=[updated_module_weights], allow_partial_loading=allow_partial_loading
            )

    @property
    def _num_kv_heads(self) -> int:
        config = self._model.config
        if isinstance(config, Qwen3VLMoeTextConfig):
            num_kv_heads = getattr(config, "num_key_value_heads", None)
            if num_kv_heads is None:
                num_kv_heads = config.num_attention_heads
        elif isinstance(config, Qwen3VLMoeVisionConfig):
            num_kv_heads = config.num_heads
        else:
            raise TypeError(
                "Expected `Qwen3VLMoeTextConfig` or `Qwen3VLMoeVisionConfig`, "
                f"got {type(config).__name__}"
            )

        return num_kv_heads
