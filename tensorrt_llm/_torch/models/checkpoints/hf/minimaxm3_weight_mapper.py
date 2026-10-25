# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

MINIMAX_M3_PARAMS_MAP = {
    r"^(.*\.block_sparse_moe)\.e_score_correction_bias$": r"\1.gate.e_score_correction_bias",
}

# Routed-expert tensors of the compressed-tensors NVFP4 MoE packaging of the
# MiniMax M3 architecture. Per projection: ``weight_packed`` (uint8, two E2M1
# values per byte), ``weight_scale`` (E4M3 per 16 elements) and the FP32
# scalars ``weight_global_scale`` / ``input_global_scale``, stored as the
# multiplier 2688 / amax (2688 = E4M3 max 448 * E2M1 max 6).
_PACKED_NVFP4_EXPERT_KEY = re.compile(
    r"^(?P<proj>.*\.block_sparse_moe\.experts\.\d+\.w[123])\."
    r"(?P<kind>weight_packed|weight_global_scale|input_global_scale)$"
)
_NVFP4_GLOBAL_SCALE_RANGE = 448.0 * 6.0


def nvfp4_weight_scale_2(weight_global_scale: Tensor) -> Tensor:
    """Return the NVFP4 MoE ``weight_scale_2`` of a stored global scale.

    compressed-tensors stores ``weight_global_scale = 2688 / amax`` (E4M3 max
    448 times E2M1 max 6); the dequantization scale is its reciprocal,
    ``amax / 2688``, computed in FP32.
    """
    return torch.reciprocal(weight_global_scale.float())


def _read_global_scale(name: str, value) -> Tensor:
    """Materialize one global scale, rejecting values that would disable an expert."""
    scale = (value if isinstance(value, Tensor) else value[...]).to(torch.float32)
    if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError(
            f"Checkpoint tensor {name} must hold one finite positive global scale, "
            f"got {scale.flatten()[:4].tolist()}."
        )
    return scale


def reciprocal_global_scale(name: str, value: Tensor) -> Tensor:
    """Validate a packed expert scale and return its FP32 modelopt divisor."""
    return nvfp4_weight_scale_2(_read_global_scale(name, value))


def translate_packed_nvfp4_experts(weights: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Rename packed NVFP4 routed-expert tensors to the NVFP4 MoE contract.

    ``weight_packed`` becomes ``weight`` with the same bytes, and the global
    scales become the ``weight_scale_2`` / ``input_scale`` divisors
    (amax / 2688) the NVFP4 MoE loader reads; ``weight_scale`` keeps its name
    and bytes. Only the scalar scales are materialized. Every other key passes
    through, including already canonical ``weight_scale_2`` / ``input_scale``,
    so a second call is a no-op.

    Raises:
        ValueError: naming the checkpoint tensor, when a global scale is not
            finite and positive, or when an expert's gate (w1) and up (w3)
            weight scales differ: the fused gate/up GEMM applies one weight
            scale per expert.
    """
    matches = {}
    for name in weights.keys():
        match = _PACKED_NVFP4_EXPERT_KEY.match(name)
        if match is not None:
            matches[name] = match
    if not matches:
        return weights

    renamed = {}
    gate_up_scales: dict[str, dict[str, tuple[str, Tensor]]] = {}
    for name in weights.keys():
        value = weights[name]
        match = matches.get(name)
        if match is None:
            renamed[name] = value
            continue
        proj, kind = match["proj"], match["kind"]
        if kind == "weight_packed":
            renamed[f"{proj}.weight"] = value
        elif kind == "weight_global_scale":
            scale_2 = nvfp4_weight_scale_2(_read_global_scale(name, value))
            renamed[f"{proj}.weight_scale_2"] = scale_2
            expert, w = proj.rsplit(".", 1)
            if w != "w2":
                gate_up_scales.setdefault(expert, {})[w] = (name, scale_2)
        else:
            renamed[f"{proj}.input_scale"] = torch.reciprocal(_read_global_scale(name, value))

    for scales in gate_up_scales.values():
        if len(scales) == 2 and not torch.equal(scales["w1"][1], scales["w3"][1]):
            raise ValueError(
                f"Checkpoint tensors {scales['w1'][0]} and {scales['w3'][0]} decode to "
                "different weight scales; the fused gate/up expert GEMM needs one."
            )
    return ConsumableWeightsDict.take_ownership(weights, renamed)


@register_mapper("HF", "MiniMaxM3SparseForCausalLM")
@register_mapper("HF", "MiniMaxM3SparseForConditionalGeneration")
class MiniMaxM3HfWeightMapper(HfWeightMapper):
    """Handle M3 gate naming, MXFP8 GQA duplication and packed NVFP4 experts."""

    def __init__(self) -> None:
        super().__init__()
        self.params_map = MINIMAX_M3_PARAMS_MAP

    def preprocess_weights(self, weights: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """See :func:`translate_packed_nvfp4_experts`."""
        return translate_packed_nvfp4_experts(weights)

    def _duplicate_kv_weights(
        self, module: nn.Module, new_name: str, weights: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if new_name not in ["k_proj", "v_proj"]:
            return weights

        duplicated_keys = ["weight", "bias"]
        quant_config = getattr(module, "quant_config", None)
        if quant_config is not None:
            quant_mode = quant_config.quant_mode
            if quant_mode.has_nvfp4():
                duplicated_keys.append("weight_scale")
            if quant_mode.has_mxfp8():
                duplicated_keys.extend(["weight_scale", "weight_scale_inv"])

        return {
            key: self._duplicate_kv(
                weight=value[:], num_kv_heads=self._num_kv_heads, tensor_parallel_size=self._tp_size
            )
            if key in duplicated_keys
            else value
            for key, value in weights.items()
        }
