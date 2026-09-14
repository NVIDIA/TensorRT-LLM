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
"""``trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_fp8``."""

import torch

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext
from ..impl_identity import register_moe_impl
from ..quantization import W4A8MXFP4FP8TRTLLMGenFusedMoEMethod
from .base import TrtllmGenFusedMoEBase
from .eligibility import check_trtllm_gen_leaf
from .identity import PROVIDER_TRTLLM, TrtllmProviderTraits, trtllm_gen_descriptor
from .kernel_inputs import prepare_kernel_inputs


@register_moe_impl
class TrtllmTrtllmGenW4a8Mxfp4Fp8Impl(TrtllmProviderTraits, TrtllmGenFusedMoEBase):
    """``trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_fp8``.

    Single-provider like the NVFP4/FP8 leaf, and likewise on its own runner op
    rather than ``self.op_backend``.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM,
        "w4a8_mxfp4_fp8",
        "TRTLLM-Gen e4m3_mxe2m1 block-scale runner: MXFP4 weights, FP8 activations.",
    )

    supports_gptoss_style = True
    needs_zero_expert_bias = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return W4A8MXFP4FP8TRTLLMGenFusedMoEMethod()

    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, pad_size))
        # Two static per-tensor scales, one per side of the fused FC1: the
        # post-communication path dequantizes with the gate-less scale.
        scale = self.fc31_input_dequant[0] if post_quant_comm else self.fc31_input_gate_dequant[0]
        x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor | tuple:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = prepare_kernel_inputs(self, ctx)

        assert k.do_finalize, (
            "e4m3_mxe2m1_block_scale_moe_runner does not support do_finalize=False"
        )
        intermediate_size_per_partition_padded = self.w3_w1_weight.shape[-2] // 2

        result = torch.ops.trtllm.e4m3_mxe2m1_block_scale_moe_runner(
            k.router_logits,
            k.routing_bias,
            k.x,
            self.w3_w1_weight,
            self.w3_w1_weight_scale,
            self.w3_w1_bias,
            self.act_alpha,
            self.act_beta,
            self.act_clamp,
            self.w2_weight,
            self.w2_weight_scale,
            self.w2_bias,
            self.fc31_input_dequant,
            self.fc31_input_gate_dequant,
            self.fc2_input_dequant,
            self.num_slots,
            k.top_k,
            k.n_group,
            k.topk_group,
            intermediate_size_per_partition_padded,
            self.hidden_size,
            self.quant_method.intermediate_size_per_partition_lean,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            0,  # act_type
            k.token_final_scales,
            k.token_selected_experts,
            output=k.moe_output,
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )
        # When output is provided, use it directly as the result
        if k.moe_output is not None:
            return k.moe_output
        return result[:, : self.hidden_size].contiguous()
