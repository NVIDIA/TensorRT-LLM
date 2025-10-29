import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.linear import W4A16_AWQ_LinearMethod

from ..base_weight_mapper import BaseWeightMapper


@register_mapper("HF")
class HfWeightMapper(BaseWeightMapper):

    def __init__(self):
        super().__init__()
        self._callbacks = [
            self._duplicate_kv_weights,
        ]

    def map_weights(self) -> None:
        self.mapping.update({
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        })

    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        module_weights = []

        for new_name in self._mapping[module_name]:
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)
            for callback in self._callbacks:
                fw = callback(module, new_name, fw)
            module_weights.append(fw)

        return module_weights

    def should_skip_module(self, module_name: str) -> bool:
        if self.model.config.tie_word_embeddings and module_name.startswith(
                "lm_head"):
            return True

        # Skip loading weights for embedding and lm_head if LoRA is enabled and has custom values
        if hasattr(self.model, "model") and hasattr(
                self.model.model, 'has_custom_embed_tokens'
        ) and self.model.model.has_custom_embed_tokens and module_name == "model.embed_tokens":
            return True
        if hasattr(
                self.model, 'has_custom_lm_head'
        ) and self.model.has_custom_lm_head and module_name == "lm_head":
            return True

        # WAR: better solution is that llama has its own load_weights function.
        if module_name.split('.')[-1] == 'next_layer_layernorm':
            return True

        return super().should_skip_module(module_name)

    def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
                              weights: dict):
        if new_name in ['k_proj', 'v_proj']:
            # k_proj and v_proj shape is [num_kv_heads*head_dim, hidden_dim]
            if isinstance(module.quant_method, W4A16_AWQ_LinearMethod):
                num_kv_heads = weights['weight'].shape[0] * 2 // self._head_dim
            else:
                num_kv_heads = weights['weight'].shape[0] // self._head_dim
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
