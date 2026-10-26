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
"""``trtllm.cutlass.grouped_gemm.w4a16``."""

import torch

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import W4A16WoqPerChannelFusedMoEMethod
from .base import CutlassFusedMoEBase, FusedMoeOpFlags
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassW4a16Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a16``.

    INT4 weight-only with per-channel scales.
    """

    descriptor = cutlass_descriptor(
        "w4a16",
        "CUTLASS grouped GEMM over INT4 per-channel weight-only, SM80+.",
    )
    sm_support = SmSupport(minimum=80)
    supported_dtypes = HP_DTYPES
    # Packed INT4 sits in int8 storage; without the quint4x2 view the op
    # would build the INT8 runner.
    FUSED_MOE_OP_FLAGS = FusedMoeOpFlags(weight_dtype=torch.quint4x2, use_woq_per_channel=True)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return W4A16WoqPerChannelFusedMoEMethod()
