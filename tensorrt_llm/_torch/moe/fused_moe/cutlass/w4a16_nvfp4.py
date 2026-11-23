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
"""``trtllm.cutlass.grouped_gemm.w4a16_nvfp4``."""

from dataclasses import replace

import torch

from ....utils import Fp4QuantizedTensor
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERunContext,
    nvfp4_fc1_row_alignment_rejection,
    require_comm_plan,
)
from ..impl_identity import register_moe_impl
from ..quantization import W4A16NVFP4CutlassFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassW4a16Nvfp4Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a16_nvfp4``.

    Weights stay NVFP4 in memory and are dequantized to the activation dtype
    every forward, so what runs is the unquantized kernel and the SM range is
    that path's.
    """

    descriptor = cutlass_descriptor(
        "w4a16_nvfp4",
        "NVFP4 weights dequantized to the activation dtype every forward, SM80+.",
    )
    sm_support = SmSupport(minimum=80)
    supported_dtypes = HP_DTYPES

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d, nvfp4_fc1_row_alignment_rejection(p, d))

    def _get_quant_method(self) -> object:
        return W4A16NVFP4CutlassFusedMoEMethod()

    def quantize_input(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        post_quant_comm: bool = True,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the AWQ pre-quant scale, keeping the activations high precision.

        This path keeps activations high precision -- ``run_moe`` dequantizes
        the NVFP4 weights and runs the unquantized kernel -- so unlike the
        ``nvfp4`` leaf there is no FP4 activation quantization here. A pre-quant
        scale is not part of that quantization, though: an AWQ-style checkpoint
        stores ``W / s`` and ``load_quant_scales`` fills ``fc31_act_scale`` from
        its ``*.pre_quant_scale`` keys regardless of ``quant_algo``, so the
        dequantized weights expect the scaled activation ``x * s``. Matches the
        dense ``W4A16NVFP4LinearMethod``, which applies ``pre_quant_scale`` the
        same way.

        See :meth:`CutlassFusedMoEBase.quantize_input` for the arguments.
        """
        del post_quant_comm, kwargs
        if self.fc31_act_scale is not None:
            assert not isinstance(x, Fp4QuantizedTensor), (
                "Fp4QuantizedTensor is not expected for AWQ quantization."
            )
            x = x * self.fc31_act_scale
        return x, None

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        """Dequantize the active experts into a static workspace, then run unquantized.

        CUDA-graph capturable: the workspace is static and the scatter stays in
        bounds because the ids are clamped to the local range.
        """
        del workspace  # Cutlass allocates its own intermediates.
        # Checked before the dequant writes the static workspace.
        require_comm_plan(self, ctx)
        if ctx.output_dtype is None:
            ctx = replace(ctx, output_dtype=ctx.x.dtype)
        # Dequantization indexes local weights; the MoE kernel receives the
        # global ids and filters non-local work itself.
        local_ids, _ = self._local_slot_ids(ctx.token_selected_experts)
        w3_w1_hp, w2_hp = self.quant_method.dequant_active_experts_to_hp(
            self, local_ids, ctx.output_dtype
        )
        return self._run_fused_moe_op(ctx, dequantized_weights=(w3_w1_hp, w2_hp))
