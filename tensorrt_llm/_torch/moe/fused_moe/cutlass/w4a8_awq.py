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
"""``trtllm.cutlass.grouped_gemm.w4a8_awq``."""

import torch

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import WInt4AFP8FusedMoEMethod
from .base import CutlassFusedMoEBase, FusedMoeOpFlags
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassW4a8AwqImpl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a8_awq``.

    INT4 group-scaled weights with FP8 activations, which the kernel quantizes
    itself.
    """

    descriptor = cutlass_descriptor(
        "w4a8_awq",
        "CUTLASS grouped GEMM over INT4 group-scaled weights with FP8 activations.",
    )
    sm_support = SmSupport(allowed=frozenset({89, 90}))
    supported_dtypes = HP_DTYPES
    requires_oss_cutlass_build = True
    FUSED_MOE_OP_FLAGS = FusedMoeOpFlags(weight_dtype=torch.quint4x2, use_w4_group_scaling=True)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return WInt4AFP8FusedMoEMethod()
