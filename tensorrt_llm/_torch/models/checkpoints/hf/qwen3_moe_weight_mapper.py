from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import \
    Qwen2MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Qwen3MoeForCausalLM")
class Qwen3MoeHfWeightMapper(Qwen2MoeHfWeightMapper):

    def should_skip_module(self, module_name: str) -> bool:
        if module_name.startswith("draft_model"):
            return True
        return super().should_skip_module(module_name)

    def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
                              weights: dict):
        tensors_to_duplicate = ["weight", "bias"]
        if module.quant_config.quant_mode.has_nvfp4():
            tensors_to_duplicate.append("weight_scale")
        if module.quant_config.quant_mode.has_fp8_block_scales():
            tensors_to_duplicate.append("weight_scale_inv")

        if new_name in ['k_proj', 'v_proj']:
            num_kv_heads_list = [self._num_kv_heads
                                 ] * len(weights) if isinstance(
                                     self._num_kv_heads,
                                     int) else self._num_kv_heads
            processed_weights = {
                k:
                self._duplicate_kv(weight=v[:],
                                   num_kv_heads=num_kv_heads_list[i],
                                   tensor_parallel_size=self._tp_size)
                if k in tensors_to_duplicate else v
                for i, (k, v) in enumerate(weights.items())
            }
            return processed_weights

        return weights
