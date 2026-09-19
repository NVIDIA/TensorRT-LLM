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
"""``flashinfer.trtllm_gen.fused_moe.none`` -- the unquantized path."""

import torch

from tensorrt_llm._torch.utils import ActivationType

from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
)
from ..impl_environment import MoEDep
from ..impl_identity import register_moe_impl
from ..interface import _reject
from ..quantization import BF16TRTLLMGenFusedMoEMethod
from ..routing import DeepSeekV3MoeRoutingMethod
from .base import TrtllmGenFusedMoEBase
from .eligibility import (
    check_flashinfer_shard_alignment,
    check_no_activation_constants,
    check_no_expert_bias,
    check_trtllm_gen_leaf,
)
from .identity import PROVIDER_FLASHINFER, trtllm_gen_descriptor
from .kernel_inputs import prepare_kernel_inputs, to_trtllm_gen_act_type


@register_moe_impl
class FlashinferTrtllmGenBf16Impl(TrtllmGenFusedMoEBase):
    """``flashinfer.trtllm_gen.fused_moe.none`` -- the unquantized path.

    FlashInfer-exclusive, and the one leaf reached without the opt-in flag:
    ``trtllm_bf16_moe`` has no counterpart in TRT-LLM's own cubins, so there is
    no native leaf to prefer and nothing for a flag to switch between. It is
    also the only leaf whose kernel does not route internally, which is what
    ``_requires_separated_routing`` below says.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER,
        "none",
        "FlashInfer's TRTLLM-Gen bf16 fused MoE, unquantized, SM100 family.",
    )

    #: What the FlashInfer bf16 kernels implement. Read from both sides here:
    #: the eligibility gate below and ``_check_configs``.
    _BF16_SUPPORTED_ACTIVATIONS = {
        ActivationType.Swiglu,
        ActivationType.Relu2,
    }

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(
            cls,
            p,
            d,
            cls._check_bf16_path(p, d),
            check_no_expert_bias(cls, p),
            check_no_activation_constants(cls, p),
        )

    def _requires_separated_routing(self) -> bool:
        """``trtllm_bf16_moe`` has no internal routing, so top-k comes from the host.

        The DeepSeekV3 exception stays a runtime test because routing is a
        per-layer argument, not part of the identity: that kernel's separated
        variant has accuracy issues, so its fused form is used instead.
        """
        return not isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod)

    def _check_configs(self) -> None:
        """The unquantized kernels take no activation constants at all.

        Restates the gate above against the constructed instance, because
        direct construction reaches here without a resolution query, and adds
        the clamp -- which ``swiglu_gptoss_style`` does not cover.
        """
        assert self.activation_type in self._BF16_SUPPORTED_ACTIVATIONS, (
            f"{type(self).__name__} only supports "
            f"{[a.name for a in self._BF16_SUPPORTED_ACTIVATIONS]} activations, "
            f"got {self.activation_type.name}."
        )
        assert (
            not self.bias
            and self.act_alpha is None
            and self.act_beta is None
            and self.act_clamp is None
        ), f"{type(self).__name__} does not support bias/swiglu custom parameters."

    @classmethod
    def _check_bf16_path(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support bias/swiglu custom parameters.",
            )
        # Same set _check_configs asserts on, so the verdict and the
        # constructor agree instead of failing later at create_weights.
        if p.activation_type not in cls._BF16_SUPPORTED_ACTIVATIONS:
            supported = ", ".join(
                sorted(activation.name for activation in cls._BF16_SUPPORTED_ACTIVATIONS)
            )
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} only supports {supported} activations, got {p.activation}",
            )
        # Stronger than MoEDep.FLASHINFER: the wheel has to expose the two
        # bf16 entry points, which older ones do not.
        if not d.env.has_dep(MoEDep.FLASHINFER_BF16_MOE):
            return _reject(
                MoERejectReason.DEP_MISSING,
                f"{cls.__name__} requires FlashInfer fused MoE with trtllm_bf16_moe support.",
            )
        # The same per-rank shard rule the quantized FlashInfer leaves run.
        # These kernels constrain only the intermediate dimension, which is
        # what passing no ``input_hidden_alignment`` says.
        return check_flashinfer_shard_alignment(cls, p, d, weight_alignment=128)

    def _get_quant_method(self) -> object:
        return BF16TRTLLMGenFusedMoEMethod()

    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor | tuple:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = prepare_kernel_inputs(self, ctx)

        result = self.op_backend.run_bf16_moe(
            router_logits=k.router_logits,
            routing_bias=k.routing_bias,
            hidden_states=k.x,
            gemm1_weights=self.w3_w1_weight,
            gemm2_weights=self.w2_weight,
            num_experts=self.num_slots,
            top_k=k.top_k,
            n_group=k.n_group,
            topk_group=k.topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            routed_scaling_factor=k.routed_scaling_factor,
            routing_method_type=self.routing_method.routing_method_type,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            gated_act_type=to_trtllm_gen_act_type(self.activation_type),
            output=k.moe_output,
            use_shuffled_weight=getattr(self.quant_method, "use_shuffled_weight", False),
            weight_layout=getattr(self.quant_method, "weight_layout", 0),
            do_finalize=k.do_finalize,
        )
        if not k.do_finalize:
            return self._unfinalized(result)
        return result
