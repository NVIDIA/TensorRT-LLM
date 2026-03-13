from typing import Union

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.utils import split
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM

# Register Qwen3_5 configs, TODO: Remove this once we have a proper transformers package
from transformers import AutoConfig, PretrainedConfig  # isort: skip

class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"
    base_config_key = "text_config"
    
    def __init__(
        self,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class Qwen3_5VisionConfig(PretrainedConfig):
    model_type = "qwen3_5"
    base_config_key = "vision_config"

class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"vision_config": Qwen3_5VisionConfig, "text_config": Qwen3_5TextConfig}
    
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

logger.warning_once(
    "transformers version below 5.1.0 does not support 'Qwen3_5Config'. "
    "Register Qwen3_5Config to mimic the Qwen3_5 model.",
    key="QWEN3_5_REGISTER_WARNING"
)
AutoConfig.register(Qwen3_5TextConfig.model_type, Qwen3_5TextConfig)
AutoConfig.register(Qwen3_5Config.model_type, Qwen3_5Config)
# End of the config register.


@register_mapper("HF", "Qwen3_5ForConditionalGeneration")
class Qwen3_5HfWeightMapper(HfWeightMapper):
    @property
    def _head_dim(self) -> int:
        config = self.model.config
        if (head_dim := getattr(config, "head_dim", None)) is not None:
            return head_dim
        if isinstance(config, Qwen3_5TextConfig):
            num_heads = config.num_attention_heads
        elif isinstance(config, Qwen3_5VisionConfig):
            num_heads = config.num_heads
        else:
            raise TypeError(f"Unexpected config class {type(config).__name__}.")

        return config.hidden_size // num_heads
    
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
        # To process the language_model part of qwen3_5
        config = self.config.pretrained_config
        tp_size = self.config.mapping.tp_size
        tp_rank = self.config.mapping.tp_rank

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
                new_weights[key] = w
            elif "conv1d" in key or "in_proj_qkv" in key:
                # Note: Qwen3-Next use in_proj_qkvz, in_proj_ba in HF ckpt,
                # and Qwen3.5 use seperate in_proj_qkv, in_proj_z, in_proj_b, in_proj_a in HF ckpt
                # Qwen3-Next and Qwen3.5 have different qkv_proj layout, 
                # we have to reorder the in_proj_qkv weight of Qwen3.5
                # Refer: https://github.com/vllm-project/vllm/blob/9e19f8338b4098047175ca3119d5ae0368bcf24a/vllm/model_executor/models/qwen3_next.py#L407
                
                w = weights[name]
                # removing dim(1) because we are using Linear to store conv1d weights
                if "conv1d.weight" in key:
                    w = w.squeeze(1)

                split_dims = [linear_key_dim, linear_key_dim, linear_value_dim]
                # NOTE: When using FP8 block quantization for in_proj_qkv,
                # we need also to reorder the corresponding conv1d weight_scale_inv
                # The split dims should be divided by 128, which is the dim 0 of weight block size (128, 128)
                if "in_proj_qkv.weight_scale_inv" in key:
                    if self.config.quant_config.quant_mode.has_fp8_block_scales():
                        split_dims = [linear_key_dim // 128, linear_key_dim // 128, linear_value_dim // 128]
                    else:
                        raise ValueError(f"Unexpected quantization {self.config.quant_config.quant_algo} for in_proj_qkv, currently only support FP8 block quantization for in_proj_qkv")
                q, k, v = torch.split(
                    w, split_dims,
                    dim=0)

                w = []
                for rank in range(tp_size):
                    q_rank = split(q, tp_size, rank)
                    k_rank = split(k, tp_size, rank)
                    v_rank = split(v, tp_size, rank)
                    y = torch.concat([q_rank, k_rank, v_rank])
                    w.append(y)
                w = torch.concat(w).contiguous()
                new_weights[key] = w
            else:
                new_weights[key] = weights[name]
        
        return new_weights
