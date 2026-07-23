from collections.abc import Mapping

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import register_mapper

from ..base_weight_mapper import BaseWeightMapper


@register_mapper("MX")
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

    def apply_callbacks(
            self, module: nn.Module, module_name: str,
            module_names_breakdown: list[str],
            weights: Mapping[str,
                             torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        module_weights = []

        # For each raw checkpoint module name needed by `module`:
        for source_name in self._mapping[module_name]:
            # Find the tensors under this checkpoint module and remove
            # their module path prefixes
            fw: dict[str, torch.Tensor] = self.filter_weights(
                '.'.join(module_names_breakdown + [source_name]), weights)
            for callback in self._callbacks:
                fw = callback(module, source_name, fw)
            module_weights.append(fw)

        return module_weights

    def should_skip_module(self, module_name: str) -> bool:
        if getattr(self.model.config, 'tie_word_embeddings',
                   False) and module_name.startswith("lm_head"):
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

    @property
    def _num_kv_heads(self) -> int:
        config = self.model.config
        if hasattr(config, 'num_key_value_heads'
                   ) and config.num_key_value_heads is not None:
            return config.num_key_value_heads
        return config.num_attention_heads

    def _duplicate_kv_weights(
            self, module: nn.Module, new_name: str,
            weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """

        If `new_name` is `k_proj` or `v_proj`, duplicate "weight" and "bias" weights
        in `weights` if needed so that they can be sliced later to match sliced query heads,
        allowing tensor parallelism to work on attention.

        Returned the potentially processed `weights`.
        """
        if new_name in ['k_proj', 'v_proj']:
            num_kv_heads = self._num_kv_heads

            duplicated_keys = ["weight", "bias"]
            if module.quant_config.quant_mode.has_nvfp4():
                duplicated_keys.append("weight_scale")

            processed_weights = {
                k:
                self._duplicate_kv(weight=v[:],
                                   num_kv_heads=num_kv_heads,
                                   tensor_parallel_size=self._tp_size)
                if k in duplicated_keys else v
                for k, v in weights.items()
            }
            return processed_weights

        return weights

    def _duplicate_kv(self, weight: torch.Tensor, num_kv_heads: int,
                      tensor_parallel_size: int) -> torch.Tensor:
        """
        Duplicate K/V proj weight to match `tensor_parallel_size`.

        It expands kv proj weight [num_kv_heads*kv_head_dim, hidden_size] into
        [num_kv_heads*rep*kv_head_dim, hidden_size], where rep = tp_size / num_kv_heads.
        As an example, if rep == 2, the weight will be converted from logically:
        [head_0_proj_matrix, head_1_proj_matrix, ...] to
        [head_0_proj_matrix, head_0_proj_matrix, head_1_proj_matrix, head_1_proj_matrix, ...]
        so that there are `tensor_parallel_size` proj matrices total, to be divided later
        as one proj matrix per tp rank.

        Return the expanded weight tensor.
        """

        if num_kv_heads >= tensor_parallel_size:
            # assume `num_kv_heads` is divisible by tp_size
            assert num_kv_heads % tensor_parallel_size == 0
            return weight

        # assume tp_size is divisible by `num_kv_heads`
        assert tensor_parallel_size % num_kv_heads == 0
        reps = tensor_parallel_size // num_kv_heads

        # bias
        if weight.ndim == 1:
            # From Yijing: this might be a bug, repeat_interleave does:
            # repeat_interleave([1, 2, 3], 2) -> [1, 1, 2, 2, 3, 3]
            # this is not how we usually slice tensors for tensor parallelism.
            # This bug is not triggered yet since most if not all of LLM attention modules
            # have no bias.
            return weight.repeat_interleave(reps)

        # weight and scale
        assert weight.shape[0] % num_kv_heads == 0
        size_per_kv_head = weight.shape[0] // num_kv_heads  # aka, kv_head_dim
        # Build a view of the tensor with shape [num_kv_heads, reps, kv_head_dim, hidden_size]
        weight = weight.reshape(num_kv_heads, size_per_kv_head,
                                -1)[:,
                                    None, :, :].expand(num_kv_heads, reps,
                                                       size_per_kv_head,
                                                       weight.shape[1])
        # Return a new tensor of shape [num_kv_heads*reps*kv_head_dim, hidden_size]
        return weight.reshape(num_kv_heads * reps * size_per_kv_head,
                              -1).clone().detach()
