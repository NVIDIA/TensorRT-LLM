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
"""``deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales``."""

import torch

from ....utils import ActivationType
from ..activation import SwigluActivation
from ..fused_moe_triton_fp8_block_scale import run_triton_fp8_block_scale_moe_fp8_act
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERunContext,
    require_comm_plan,
)
from ..impl_identity import register_moe_impl
from ..quantization import DeepSeekFP8BlockScalesFusedMoEMethod
from .base import CutlassFusedMoEBase, FusedMoeOpFlags
from .eligibility import BF16_ONLY, SmSupport, check_cutlass_leaf
from .identity import (
    KERNEL_HOPPER_GROUPED_GEMM,
    PROVIDER_DEEPGEMM,
    TECHNIQUE_CUDA,
    cutlass_descriptor,
)


@register_moe_impl
class DeepgemmCudaHopperFp8BlockScalesImpl(CutlassFusedMoEBase):
    """``deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales``.

    FP8 block scales on SM90. ``use_deepseek_fp8_block_scale=True`` makes
    ``CutlassMoeFCRunner`` run FC1/FC2 through
    ``cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/``, which is
    built on TRT-LLM's DeepGEMM port (``cpp/include/tensorrt_llm/deep_gemm``),
    hence the identity; permute, activation and finalize stay in the CUTLASS
    runner, hence the family. The runner quantizes activations itself. Only
    ``kernel_name`` separates this identity from the SM100/103/107 leaf in
    ``..fused_moe_deepgemm``.

    Decode-sized batches (at most ``TRITON_FP8_ACT_MAX_TOKENS`` tokens on this
    rank) with a plain SwiGLU run Triton FP8-activation kernels instead: the
    grouped GEMM walks every expert on one persistent CTA per SM, while the
    Triton kernels launch only the expert blocks the batch routes to.
    """

    #: Largest batch (tokens on this rank) that runs the Triton FP8-activation path.
    TRITON_FP8_ACT_MAX_TOKENS = 32

    descriptor = cutlass_descriptor(
        "fp8_block_scales",
        "DeepGEMM grouped GEMM over FP8 block scales, driven by CutlassMoeFCRunner, SM90.",
        provider=PROVIDER_DEEPGEMM,
        technique=TECHNIQUE_CUDA,
        kernel_name=KERNEL_HOPPER_GROUPED_GEMM,
    )
    sm_support = SmSupport(allowed=frozenset({90}))
    #: The block-scale GEMM runner only has BF16 A / output instantiations.
    supported_dtypes = BF16_ONLY
    FUSED_MOE_OP_FLAGS = FusedMoeOpFlags(use_deepseek_fp8_block_scale=True)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return DeepSeekFP8BlockScalesFusedMoEMethod()

    def _triton_fp8_act_swiglu(self, ctx: MoERunContext) -> tuple[float | None, bool] | None:
        """SwiGLU clamp settings when this call can take the Triton FP8-activation path.

        Returns ``(limit, clamp_after_silu)``, or None to keep the fused op: a
        plain SwiGLU with at most a scalar clamp, no expert bias, BF16 input, a
        decode-sized batch, and no LoRA or caller-provided output buffer.
        """
        activation = self.activation
        if (
            self.activation_type != ActivationType.Swiglu
            or not isinstance(activation, SwigluActivation)
            or self.w3_w1_bias is not None
            or self.w2_bias is not None
            or ctx.x_sf is not None
            or ctx.x.dtype != torch.bfloat16
            or ctx.x.shape[0] > self.TRITON_FP8_ACT_MAX_TOKENS
            or ctx.lora_params is not None
            or ctx.comm_plan.moe_output is not None
        ):
            return None
        limit = activation.clamp
        if isinstance(limit, torch.Tensor):
            return None
        return (None if limit is None else float(limit), activation.clamp_after_silu)

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # Both paths allocate their own intermediates.
        require_comm_plan(self, ctx)
        swiglu = self._triton_fp8_act_swiglu(ctx)
        if swiglu is None:
            return self._run_fused_moe_op(ctx)

        # The scheduler leaves token_final_scales as None when the router
        # weights were already folded into x.
        token_final_scales = ctx.token_final_scales
        if token_final_scales is None:
            token_final_scales = torch.ones_like(ctx.token_selected_experts, dtype=torch.float32)
        if self.ep_size == 1:
            # Every routed slot is local: the IDs and weights need no remap.
            local_ids, local_scales = ctx.token_selected_experts, token_final_scales
        else:
            # The kernel indexes local experts; zero the scale of every pair
            # this rank does not own.
            local_ids, offsets = self._local_slot_ids(ctx.token_selected_experts)
            local_scales = token_final_scales * (local_ids == offsets).to(token_final_scales.dtype)

        swiglu_limit, swiglu_clamp_after_silu = swiglu
        return run_triton_fp8_block_scale_moe_fp8_act(
            ctx.x,
            local_ids,
            local_scales,
            self.w3_w1_weight,
            self.quant_scales.fc_weight_scales,
            self.w2_weight,
            self.quant_scales.proj_weight_scales,
            swiglu_limit=swiglu_limit,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
            output_dtype=ctx.output_dtype,
        )
