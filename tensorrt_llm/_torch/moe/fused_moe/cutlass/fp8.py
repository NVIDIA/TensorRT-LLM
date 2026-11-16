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
"""``trtllm.cutlass.grouped_gemm.fp8``."""

import torch

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import FP8QDQFusedMoEMethod
from .base import CutlassFusedMoEBase, quantize_input_static_e4m3
from .eligibility import HP_DTYPES_WITH_FP32, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor
from .lora import CutlassMoELoraMixin


@register_moe_impl
class TrtllmCutlassFp8Impl(CutlassMoELoraMixin, CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.fp8``.

    Per-tensor FP8, quantized and dequantized around the GEMM rather than
    block-scaled. One of the two formats the op fuses routed-expert LoRA for.
    """

    descriptor = cutlass_descriptor(
        "fp8",
        "CUTLASS grouped GEMM over per-tensor FP8 (qdq) weights, SM89+.",
        supports_moe_lora=True,
    )
    sm_support = SmSupport(minimum=89)
    supported_dtypes = HP_DTYPES_WITH_FP32
    LORA_OP_INPUT_DTYPE = torch.float8_e4m3fn
    quantize_input = quantize_input_static_e4m3

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return FP8QDQFusedMoEMethod()
