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
"""``trtllm.cutlass.grouped_gemm.w4a8_mxfp4_fp8``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import W4A8MXFP4FP8CutlassFusedMoEMethod
from .base import CutlassFusedMoEBase, quantize_input_static_e4m3
from .eligibility import HP_DTYPES_WITH_FP32, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassW4a8Mxfp4Fp8Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a8_mxfp4_fp8``.

    Activations are quantized to e4m3 per tensor before the GEMM, the same step
    the per-tensor FP8 leaf takes.
    """

    descriptor = cutlass_descriptor(
        "w4a8_mxfp4_fp8",
        "CUTLASS grouped GEMM over MXFP4 weights with static FP8 activations.",
    )
    sm_support = SmSupport(allowed=frozenset({100, 103, 107}))
    supported_dtypes = HP_DTYPES_WITH_FP32
    supports_gptoss_style = True
    quantize_input = quantize_input_static_e4m3

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return W4A8MXFP4FP8CutlassFusedMoEMethod()
