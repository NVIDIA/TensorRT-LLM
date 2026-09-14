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
"""``trtllm.trtllm_gen.fused_moe.w4a8_nvfp4_fp8``."""

from typing import Optional, Union

import torch

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext
from ..impl_identity import register_moe_impl
from ..quantization import W4A8NVFP4FP8TRTLLMGenFusedMoEMethod
from .base import TrtllmGenFusedMoEBase
from .eligibility import check_trtllm_gen_leaf
from .identity import PROVIDER_TRTLLM, TrtllmProviderTraits, trtllm_gen_descriptor
from .kernel_inputs import prepare_kernel_inputs


@register_moe_impl
class TrtllmTrtllmGenW4a8Nvfp4Fp8Impl(TrtllmProviderTraits, TrtllmGenFusedMoEBase):
    """``trtllm.trtllm_gen.fused_moe.w4a8_nvfp4_fp8``.

    Single-provider, so there is no per-quant parent, and it calls its runner
    op directly instead of going through ``self.op_backend``: the FlashInfer
    wheel has no equivalent.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM,
        "w4a8_nvfp4_fp8",
        "TRTLLM-Gen fp8_fp4 block-scale runner: NVFP4 weights, FP8 activations.",
    )

    needs_zero_expert_bias = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def _check_configs(self) -> None:
        """No FC bias, no SwiGLU constants, no clamp: the runner takes none.

        ``fp8_fp4_block_scale_moe_runner`` has no parameter for any of them and
        pins ``act_type`` to plain SwiGLU, so anything the carrier materialized
        would be dropped on the floor rather than applied. Asserted here because
        the resolution gates do not cover it: ``supports_gptoss_style`` only
        sees the bias-plus-alpha-beta package as a whole, while a checkpoint can
        carry a plain clamp, or a plain bias, on its own.
        """
        assert (
            not self.bias
            and self.act_alpha is None
            and self.act_beta is None
            and self.act_clamp is None
        ), (
            f"{type(self).__name__} takes no expert bias, no swiglu alpha/beta "
            f"constants, and no activation clamp."
        )

    def _get_quant_method(self):
        return W4A8NVFP4FP8TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, 1.0 / self.fc31_input_scale
        )
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = prepare_kernel_inputs(self, ctx)

        outputs = torch.ops.trtllm.fp8_fp4_block_scale_moe_runner(
            k.router_logits,
            k.routing_bias,
            k.x,
            self.w3_w1_weight,
            self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
            self.w2_weight,
            self.w2_weight_scale.view(torch.float8_e4m3fn),
            self.fc31_scale_c.data,
            self.fc31_alpha.data,
            self.fc2_alpha.data,
            self.num_slots,
            k.top_k,
            k.n_group,
            k.topk_group,
            self.intermediate_size_per_partition,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            do_finalize=k.do_finalize,
            act_type=0,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            output=k.moe_output,
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )

        if not k.do_finalize:
            return self._unfinalized(outputs)
        # When output is provided, use it directly as the result
        return k.moe_output if k.moe_output is not None else outputs[0]
