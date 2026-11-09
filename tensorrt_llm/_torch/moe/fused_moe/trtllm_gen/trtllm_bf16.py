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
"""``trtllm.trtllm_gen.fused_moe.none`` -- the unquantized path."""

import os

import torch

from tensorrt_llm._torch.moe.expert_statistic import ExpertStatistic
from tensorrt_llm._torch.route_capture import get_active_route_capture
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator
from tensorrt_llm.tools.layer_wise_benchmarks.calibrator import Mode as CalibratorMode

from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
)
from ..impl_identity import register_moe_impl
from ..interface import _reject
from ..quantization import BF16TRTLLMGenFusedMoEMethod
from .base import TrtllmGenFusedMoEBase
from .eligibility import (
    check_intermediate_shard_alignment,
    check_no_activation_constants,
    check_no_expert_bias,
    check_trtllm_gen_leaf,
)
from .identity import PROVIDER_TRTLLM, trtllm_gen_descriptor
from .kernel_inputs import prepare_kernel_inputs, to_trtllm_gen_act_type


@register_moe_impl
class TrtllmTrtllmGenBf16Impl(TrtllmGenFusedMoEBase):
    """``trtllm.trtllm_gen.fused_moe.none`` -- the unquantized path.

    BF16 activations times BF16 weights on the ``Bmm_Bfloat16_Bfloat16Bfloat16``
    cubins, through ``trtllm::bf16_moe_runner``. The weights are stored in the
    shuffled BlockMajorK layout ``BF16TRTLLMGenFusedMoEMethod`` produces.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM,
        "none",
        "TRTLLM-Gen batched-GEMM cubins, BF16 activations and weights, SM100 family.",
    )

    #: The gemm1 epilogues the BF16 cubins ship: gated SwiGLU and non-gated
    #: Relu2. Read from both sides here: the eligibility gate below and
    #: ``_check_configs``.
    _BF16_SUPPORTED_ACTIVATIONS = {
        ActivationType.Swiglu,
        ActivationType.Relu2,
    }
    #: Every BF16 cubin has tileM 128 and tileK 128 (or 64). The hidden size is
    #: gemm1's K and gemm2's M; each rank's intermediate shard is gemm2's K and
    #: (half of) gemm1's M. Both must be multiples of this, or no cubin applies.
    _SHAPE_ALIGNMENT = 128
    #: The routing kernels' limits (``runner.cu``, ``RoutingCustomPolicy.cuh``).
    _MAX_ROUTING_EXPERTS = 2048
    _MAX_ROUTING_TOP_K = 32

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
        """Route on the host while something reads the host top-k.

        The kernel routes from the logits, which leaves Router Replay capture,
        ExpertStatistic, the layer-wise benchmark calibrator and
        ``ENABLE_PERFECT_ROUTER=expert_ids`` with no top-k to read. Asked per
        forward, because Router Replay registers its capturer after the model
        is built.
        """
        return (
            get_active_route_capture() is not None
            or ExpertStatistic.get() is not None
            or get_calibrator().mode != CalibratorMode.NONE
            # Read like MoE._init_perfect_router; the leaf does not run that init.
            or os.environ.get("ENABLE_PERFECT_ROUTER") == "expert_ids"
        )

    def supports_unfinalized_output(self) -> bool:
        """``bf16_moe_runner`` honors ``do_finalize=False`` and returns the triple."""
        return True

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
        # Rejected here rather than in the op, so the layer falls back to
        # another backend instead of failing at warmup.
        if p.hidden_size is not None and p.hidden_size % cls._SHAPE_ALIGNMENT != 0:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"{cls.__name__} requires hidden_size ({p.hidden_size}) to be a "
                f"multiple of {cls._SHAPE_ALIGNMENT}",
            )
        if d.num_slots > cls._MAX_ROUTING_EXPERTS:
            return _reject(
                MoERejectReason.ROUTING_UNSUPPORTED,
                f"{cls.__name__} routes at most {cls._MAX_ROUTING_EXPERTS} experts, "
                f"got {d.num_slots} slots",
            )
        if p.top_k is not None and p.top_k > cls._MAX_ROUTING_TOP_K:
            return _reject(
                MoERejectReason.ROUTING_UNSUPPORTED,
                f"{cls.__name__} routes at most top_k={cls._MAX_ROUTING_TOP_K}, got {p.top_k}",
            )
        return check_intermediate_shard_alignment(cls, p, d, weight_alignment=cls._SHAPE_ALIGNMENT)

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
            do_finalize=k.do_finalize,
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )
        if not k.do_finalize:
            return self._unfinalized(result)
        return result
