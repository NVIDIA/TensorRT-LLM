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
"""Native MXFP8 execution for DeepSeek-V4.1's 32x32-scaled Engram WKV."""

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..linear import Linear, MXFP8LinearMethod


def expand_engram_scales(scale: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Expand UE8M0 ``[ceil(N/32), K/32]`` tiles into native ``[N, K/32]`` scales.

    Only scale metadata is repeated: the E4M3 weight bytes remain unchanged, so
    this conversion introduces no additional weight quantization error.
    """
    expected = ((out_features + 31) // 32, in_features // 32)
    if in_features % 32 or scale.shape != expected:
        raise ValueError(f"Engram WKV requires 32x32 scales of shape {expected}, got {scale.shape}")
    if scale.dtype == torch.float8_e8m0fnu:
        scale = scale.view(torch.uint8)
    elif scale.dtype != torch.uint8:
        raise ValueError(f"Engram WKV scales must be UE8M0 bytes, got {scale.dtype}")
    return scale.repeat_interleave(32, dim=0)[:out_features].contiguous()


class EngramFp8Projection(Linear):
    """Replicated WKV using the native MXFP8 GEMM and existing backend tuning.

    Weights retain the report's E4M3/UE8M0 values. The loader expands the 32-row
    scale tiles, then the shared linear method interleaves them for Blackwell.
    Activations are dynamically quantized per 32 lanes. On devices without the
    compiled MXFP8 backend, the linear method supplies its dequantized reference.

    The engine tunes native prefill and FlashInfer decode separately. At the
    released ``[25600, 6144]`` shape, the tuned FlashInfer graph path avoids the
    native kernel's low-token utilization penalty on GB300. An explicit
    ``TRTLLM_MXFP8_GEMM_BACKEND`` setting still takes precedence.
    """

    # Consumed by the engine's existing MXFP8 warmup/capture lifecycle. Keep
    # standalone and eager execution native; FlashInfer is selected only after
    # startup tuning and inside the decode graph capture context.
    _use_flashinfer_mxfp8_decode_graph_default = True

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=False,
            dtype=dtype,
            quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
            reduce_output=False,
        )

    def create_weights(self) -> None:
        """Retain WKV's explicit checkpoint format through model quant policies.

        The model's generic ``*engram*`` exclusion is intended for its table,
        norms and legacy BF16 projection. The shared post-init walk resets the
        quant config and recreates weights on excluded Linear modules. This
        specialized FP8 consumer must restore its format before that recreation,
        or its loader would silently feed unscaled E4M3 values to BF16 GEMM.
        """
        self.quant_config = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
        super().create_weights()

    def load_weights(self, weights: list[dict]) -> None:
        """Load one replicated ``weight`` and checkpoint ``scale`` pair."""
        if not isinstance(self.quant_method, MXFP8LinearMethod):
            raise RuntimeError("Engram FP8 WKV requires MXFP8LinearMethod to consume its scales")
        if len(weights) != 1 or "weight" not in weights[0] or "scale" not in weights[0]:
            raise ValueError("Engram FP8 WKV requires exactly one weight/scale checkpoint pair")
        weight = weights[0]["weight"][:]
        scale = weights[0]["scale"][:]
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError("Engram WKV weight shape does not match the projection")
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Engram FP8 WKV requires E4M3 checkpoint weights")
        expanded_scale = expand_engram_scales(scale, self.out_features, self.in_features)
        super().load_weights([{"weight": weight, "weight_scale": expanded_scale}])
