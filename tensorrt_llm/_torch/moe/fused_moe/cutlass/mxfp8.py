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
"""``trtllm.cutlass.grouped_gemm.mxfp8``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import MXFP8CutlassFusedMoEMethod
from .base import CutlassFusedMoEBase, FusedMoeOpFlags, quantize_input_mxfp8
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassMxfp8Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.mxfp8``.

    W8A8 e4m3 x e4m3 with UE8M0 1x32 block scales on both sides. Shares the
    MXFP4 family's activation quantization; only the weight element widens from
    fp4 to fp8.
    """

    descriptor = cutlass_descriptor(
        "mxfp8",
        "CUTLASS grouped GEMM over MXFP8 weights and dynamic MXFP8 activations.",
    )
    # The SM100-family block-scaled kernels, which include SM107.
    sm_support = SmSupport(allowed=frozenset({100, 103, 107}))
    supported_dtypes = HP_DTYPES
    supports_gptoss_style = True
    requires_oss_cutlass_build = True
    FUSED_MOE_OP_FLAGS = FusedMoeOpFlags(use_mxfp8_act_scaling=True, use_mxfp8_weight_scaling=True)
    quantize_input = quantize_input_mxfp8

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return MXFP8CutlassFusedMoEMethod()
