from typing import Union

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import \
    Qwen2MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_nemotron_h import split
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM


@register_mapper("HF", "Qwen3NextForCausalLM")
class Qwen3NextHfWeightMapper(Qwen2MoeHfWeightMapper):

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

    def preprocess_weights(self, weights: dict) -> dict:
        config = self.config.pretrained_config
        tp_size = self.config.mapping.tp_size
        tp_rank = self.config.mapping.tp_rank

        # linear_num_value_heads = config.linear_num_value_heads
        # linear_num_key_heads = config.linear_num_key_heads
        # linear_key_head_dim = config.linear_key_head_dim
        # linear_value_head_dim = config.linear_value_head_dim
        linear_key_dim = config.linear_key_head_dim * config.linear_num_key_heads  # 16 * 128
        linear_value_dim = config.linear_value_head_dim * config.linear_num_value_heads  # 32 * 128

        new_weights = {}
        for name, _ in weights.items():
            key = name

            if "A_log" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "dt_bias" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "in_proj" in key:
                # Don't need to split in_proj weight based on the implementation of reference.
                # Need to know the reason.
                new_weights[key] = weights[name]
            elif "conv1d" in key:
                w = weights[name]
                # removing dim(1) because we are using Linear to store conv1d weights
                if "weight" in key:
                    w = w.squeeze(1)

                conv_q, conv_k, conv_v = torch.split(
                    w, [linear_key_dim, linear_key_dim, linear_value_dim],
                    dim=0)

                w = []
                for rank in range(tp_size):
                    conv_q_rank = split(conv_q, tp_size, rank)
                    conv_k_rank = split(conv_k, tp_size, rank)
                    conv_v_rank = split(conv_v, tp_size, rank)
                    y = torch.concat([conv_q_rank, conv_k_rank, conv_v_rank])
                    w.append(y)
                w = torch.concat(w).contiguous()
                new_weights[key] = w
            else:
                new_weights[key] = weights[name]

        return new_weights
