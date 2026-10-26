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
"""``trtllm.cutlass.grouped_gemm.w4a16_mxfp4``."""

from dataclasses import replace

import torch

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext
from ..impl_identity import register_moe_impl
from ..quantization import WFP4A16FusedMoEMethod
from .base import CutlassFusedMoEBase, FusedMoeOpFlags
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassW4a16Mxfp4Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a16_mxfp4``.

    SM90 only; the weight method raises elsewhere.
    """

    descriptor = cutlass_descriptor(
        "w4a16_mxfp4",
        "CUTLASS grouped GEMM over MXFP4 weights with high-precision activations, SM90.",
    )
    sm_support = SmSupport(allowed=frozenset({90}))
    supported_dtypes = HP_DTYPES
    supports_gptoss_style = True
    requires_oss_cutlass_build = True
    FUSED_MOE_OP_FLAGS = FusedMoeOpFlags(weight_dtype=torch.uint8, use_w4_group_scaling=True)
    pad_hidden_and_intermediate_to = 128

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return WFP4A16FusedMoEMethod()

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # Cutlass allocates its own intermediates.
        # Pad here rather than in ``quantize_input``, so dispatch sends unpadded
        # tensors and does not overallocate the NVLink workspace.
        pad_size = self.hidden_size - ctx.x.shape[-1]
        if pad_size > 0:
            ctx = replace(ctx, x=torch.nn.functional.pad(ctx.x, (0, pad_size)))
        return self._run_fused_moe_op(ctx)
