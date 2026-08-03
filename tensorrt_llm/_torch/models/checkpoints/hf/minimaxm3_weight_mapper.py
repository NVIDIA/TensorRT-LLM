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

from torch import Tensor, nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

MINIMAX_M3_PARAMS_MAP = {
    r"^(.*\.block_sparse_moe)\.e_score_correction_bias$": r"\1.gate.e_score_correction_bias",
}


@register_mapper("HF", "MiniMaxM3SparseForCausalLM")
@register_mapper("HF", "MiniMaxM3SparseForConditionalGeneration")
class MiniMaxM3HfWeightMapper(HfWeightMapper):
    """Handle M3 gate naming and MXFP8 GQA duplication for loader v2."""

    def __init__(self) -> None:
        super().__init__()
        self.params_map = MINIMAX_M3_PARAMS_MAP

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
