# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm.logger import logger

from ..base_weight_loader import ConsumableWeightsDict
from ..base_weight_mapper import BaseWeightMapper

_COMPRESSED_TENSORS_NVFP4_SUFFIXES = (".weight_packed", ".weight_global_scale",
                                      ".input_global_scale")


def _invert_compressed_tensors_global_scale(value) -> torch.Tensor:
    """Convert a compressed-tensors global scale to TRT-LLM's divisor form.

    A global scale of 0 marks an expert that saw no calibration data. It is
    kept as 0 rather than inverted to Inf, which would otherwise win the MoE
    per-layer max-reduction over experts and corrupt every expert's
    input_scale; a 0 entry is harmless there and silences the dead expert.
    """
    t = value if isinstance(value, torch.Tensor) else value[...]
    t = t.to(torch.float32)
    out = torch.where(t > 0,
                      1.0 / torch.clamp(t, min=torch.finfo(torch.float32).tiny),
                      torch.zeros_like(t))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).contiguous()


def _read_compressed_tensors_global_scale(name: str, value, zero_scales: list,
                                          invalid_scales: list) -> torch.Tensor:
    """Read a global scale and record whether it has entries that cannot be inverted."""
    t = value if isinstance(value, torch.Tensor) else value[...]
    t32 = t.to(torch.float32)
    if bool((~torch.isfinite(t32) | (t32 < 0)).any()):
        invalid_scales.append(name)
    elif bool((t32 == 0).any()):
        zero_scales.append(name)
    return t


def _warn_uninvertible_global_scales(zero_scales: list,
                                     invalid_scales: list) -> None:
    """Log one summary line for the global scales that were set to 0."""
    if not zero_scales and not invalid_scales:
        return
    parts = []
    if invalid_scales:
        parts.append(
            f"{len(invalid_scales)} with negative or non-finite values, which "
            f"means a broken checkpoint (e.g. {', '.join(invalid_scales[:3])})")
    if zero_scales:
        parts.append(
            f"{len(zero_scales)} with 0, as stored for experts without "
            f"calibration data (e.g. {', '.join(zero_scales[:3])})")
    logger.warning(
        "compressed-tensors NVFP4 global scales that cannot be inverted: " +
        "; ".join(parts) + ". They are set to 0; a weight global scale of 0 "
        "zeroes that expert's output.")


def convert_compressed_tensors_nvfp4_weights(
        weights: Mapping[str, Any]) -> Mapping[str, Any]:
    """Rename llm-compressor "compressed-tensors" NVFP4 tensors to TRT-LLM's.

    compressed-tensors stores
      weight_packed        (uint8, packed FP4)
      weight_scale         (FP8 per-group)
      weight_global_scale  (FP32 scalar = (FP8_MAX*FP4_MAX)/amax_weight)
      input_global_scale   (FP32 scalar = (FP8_MAX*FP4_MAX)/amax_input)
    while TRT-LLM's NVFP4 Linear and MoE loaders expect
      weight               (uint8 view of float4_e2m1x2)
      weight_scale         (FP8, same)
      weight_scale_2       (FP32 scalar = amax_weight/(FP8_MAX*FP4_MAX))
      input_scale          (FP32 scalar = amax_input/(FP8_MAX*FP4_MAX))
    so the global scales are inverted on the way through.

    Args:
        weights: checkpoint weights keyed by checkpoint name; a plain dict or
            a ``ConsumableWeightsDict``.

    Returns:
        ``weights`` itself when it holds no compressed-tensors NVFP4 names.
        Otherwise a converted mapping of the same kind; other keys are passed
        through unchanged and the caller must not use ``weights`` afterwards
        (see ``ConsumableWeightsDict.take_ownership``).

    Raises:
        ValueError: two checkpoint tensors map to the same TRT-LLM name (for
            example both ``x.weight_packed`` and ``x.weight``), which would
            otherwise keep whichever came last.
    """
    if not any(
            name.endswith(_COMPRESSED_TENSORS_NVFP4_SUFFIXES)
            for name in weights.keys()):
        return weights

    converted, sources = {}, {}
    zero_scales, invalid_scales = [], []
    for name, value in weights.items():
        source = name
        if name.endswith(".weight_packed"):
            name = name[:-len(".weight_packed")] + ".weight"
        elif name.endswith((".weight_global_scale", ".input_global_scale")):
            value = _read_compressed_tensors_global_scale(
                name, value, zero_scales, invalid_scales)
            if name.endswith(".weight_global_scale"):
                name = name[:-len(".weight_global_scale")] + ".weight_scale_2"
            else:
                name = name[:-len(".input_global_scale")] + ".input_scale"
            value = _invert_compressed_tensors_global_scale(value)
        if name in converted:
            raise ValueError(
                f"compressed-tensors NVFP4 checkpoint maps both {sources[name]!r} "
                f"and {source!r} to {name!r}")
        converted[name] = value
        sources[name] = source
    _warn_uninvertible_global_scales(zero_scales, invalid_scales)
    return ConsumableWeightsDict.take_ownership(weights, converted)


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
            assert weight.shape[0] % num_kv_heads == 0
            size_per_kv_head = weight.shape[0] // num_kv_heads
            return weight.reshape(num_kv_heads,
                                  size_per_kv_head).repeat_interleave(
                                      reps, dim=0).reshape(-1)

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
