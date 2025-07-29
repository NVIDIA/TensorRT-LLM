from typing import Union

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import \
    Qwen2MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM


@register_mapper("HF", "Qwen3MoeForCausalLM")
class Qwen3MoeHfWeightMapper(Qwen2MoeHfWeightMapper):

    def init_model_and_config(self, model: Union[nn.Module,
                                                 DecoderModelForCausalLM],
                              config: ModelConfig):
        super().init_model_and_config(model, config)
        self._num_kv_heads = model.config.num_key_value_heads if hasattr(
            model.config, 'num_key_value_heads'
        ) and model.config.num_key_value_heads is not None else model.config.num_attention_heads

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

    def _duplicate_kv(self, weight: torch.Tensor, num_kv_heads: int,
                      tensor_parallel_size: int):

        if num_kv_heads >= tensor_parallel_size:
            assert num_kv_heads % tensor_parallel_size == 0
            return weight

        assert tensor_parallel_size % num_kv_heads == 0
        reps = tensor_parallel_size // num_kv_heads

        # bias
        if weight.ndim == 1:
            return weight.repeat_interleave(reps)

        # weight and scale
        assert weight.shape[0] % num_kv_heads == 0
        size_per_kv_head = weight.shape[0] // num_kv_heads
        weight = weight.reshape(num_kv_heads, size_per_kv_head,
                                -1)[:,
                                    None, :, :].expand(num_kv_heads, reps,
                                                       size_per_kv_head,
                                                       weight.shape[1])
        return weight.reshape(num_kv_heads * reps * size_per_kv_head,
                              -1).clone().detach()
