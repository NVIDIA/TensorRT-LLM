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
"""``trtllm.triton.grouped_gemm.fp8_block_scales``."""

import torch

from ....utils import ActivationType
from ..activation import MoEActivationSupport
from ..fused_moe_triton_fp8_block_scale import run_triton_fp8_block_scale_moe
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERunContext,
    require_comm_plan,
)
from ..impl_identity import register_moe_impl
from ..quantization import DeepSeekFP8BlockScalesFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import BF16_ONLY, SmSupport, check_cutlass_leaf
from .identity import TECHNIQUE_TRITON, cutlass_descriptor


@register_moe_impl
class TrtllmTritonFp8BlockScalesImpl(CutlassFusedMoEBase):
    """``trtllm.triton.grouped_gemm.fp8_block_scales``.

    FP8 block scales on SM120, where CUTLASS TMA fails for large token counts
    (``cuTensorMapEncodeTiled`` limitations). The whole MoE runs in
    ``run_triton_fp8_block_scale_moe`` and never enters C++; the leaf takes
    only construction and the layer-format check from ``CutlassFusedMoEBase``.
    """

    descriptor = cutlass_descriptor(
        "fp8_block_scales",
        "Internal Triton FP8 block-scale grouped GEMM, SM120.",
        technique=TECHNIQUE_TRITON,
    )
    sm_support = SmSupport(allowed=frozenset({120}))
    #: The Triton kernel is instantiated for BF16 activations only.
    supported_dtypes = BF16_ONLY
    # The kernel takes no activation constants and runs only these gated kinds.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Geglu})
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return DeepSeekFP8BlockScalesFusedMoEMethod()

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        # Unlike the rest of the family, run_moe returns the kernel's own tensor.
        return False

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # The Triton kernel allocates its own intermediates.
        require_comm_plan(self, ctx)

        # The scheduler leaves token_final_scales as None when the router
        # weights were already folded into x; ones make the per-token scaling a
        # no-op rather than a missing operand.
        token_final_scales = ctx.token_final_scales
        if token_final_scales is None:
            token_final_scales = torch.ones_like(ctx.token_selected_experts, dtype=torch.float32)

        # The kernel indexes local experts, but the ids are global slots with or
        # without alltoall: dispatch keeps the global numbering and marks
        # padding with an id no rank owns (-1, or num_slots for DeepEP low
        # latency). Shift them into this rank's range and zero the scale of any
        # pair this rank does not own.
        local_n = self.expert_size_per_partition
        ids = ctx.token_selected_experts - self.slot_start
        is_local = (ids >= 0) & (ids < local_n)
        local_ids = ids.clamp(0, local_n - 1)
        local_scales = token_final_scales * is_local.to(token_final_scales.dtype)

        return run_triton_fp8_block_scale_moe(
            ctx.x,
            local_ids,
            local_scales,
            self.w3_w1_weight,
            self.quant_scales.fc_weight_scales,
            self.w2_weight,
            self.quant_scales.proj_weight_scales,
            activation_type=self.activation_type,
            output_dtype=ctx.output_dtype,
        )
