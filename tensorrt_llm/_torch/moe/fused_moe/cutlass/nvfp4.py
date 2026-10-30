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
"""``trtllm.cutlass.grouped_gemm.nvfp4``."""

import torch

from ....utils import Fp4QuantizedTensor
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERunContext,
    nvfp4_fc1_row_alignment_rejection,
)
from ..impl_identity import register_moe_impl
from ..quantization import NVFP4CutlassFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES_WITH_FP8, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor

_FP8_MAX = 448.0
_E2M1_MAX = 6.0


@register_moe_impl
class TrtllmCutlassNvfp4Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.nvfp4``.

    W4A4: ``quantize_input`` quantizes the activations to FP4 as well. Admits
    the NVFP4_AWQ / NVFP4_ARC calibration aliases, which
    ``MoEProblem.identity_quant`` folds onto ``nvfp4``.
    """

    descriptor = cutlass_descriptor(
        "nvfp4",
        "CUTLASS grouped GEMM over NVFP4 weights and activations, Blackwell.",
    )
    sm_support = SmSupport(allowed=frozenset({100, 103, 107, 120, 121}))
    supported_dtypes = HP_DTYPES_WITH_FP8
    supports_gptoss_style = True
    rejects_gptoss_expert_bias = True
    uses_dynamic_fc2_scale = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d, nvfp4_fc1_row_alignment_rejection(p, d))

    def _get_quant_method(self) -> object:
        return NVFP4CutlassFusedMoEMethod()

    def quantize_input(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        post_quant_comm: bool = True,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the AWQ pre-quant scale, then quantize to NVFP4.

        See :meth:`CutlassFusedMoEBase.quantize_input` for the arguments.
        """
        del kwargs
        if self.fc31_act_scale is not None:
            assert not isinstance(x, Fp4QuantizedTensor), (
                "Fp4QuantizedTensor is not expected for AWQ quantization."
            )
            x = x * self.fc31_act_scale

        if self.force_dynamic_quantization and hasattr(self, "fc31_weight_scale_2"):
            # Update the scales in place: captured CUDA graphs hold their addresses.
            dyn_input_scale = _FP8_MAX * _E2M1_MAX / torch.amax(torch.abs(x)).float()
            self.fc31_alpha.data.copy_(self.fc31_weight_scale_2.data / dyn_input_scale)
            self.fc31_input_scale.data.copy_(dyn_input_scale)

        if post_quant_comm:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, (
                    "Fp4QuantizedTensor should not be swizzled before communication"
                )
                x, x_sf = x.fp4_tensor, x.scaling_factor
                x_row = x.shape[0]
            else:
                x_row = x.shape[0]
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False, False
                )
            # Communication moves the scales as one 2-D row per token.
            if x_sf is not None:
                x_sf = x_sf.view((x_row, -1))
            return x, x_sf

        if isinstance(x, Fp4QuantizedTensor):
            return x, None
        return torch.ops.trtllm.fp4_quantize(
            x, self.fc31_input_scale, self.scaling_vector_size, False, True
        )

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # Cutlass allocates its own intermediates.
        dynamic_fc2_scale = None
        if self.force_dynamic_quantization and hasattr(self, "fc2_weight_scale_2"):
            dynamic_fc2_scale = self.fc2_weight_scale_2
        return self._run_fused_moe_op(ctx, dynamic_fc2_scale=dynamic_fc2_scale)
