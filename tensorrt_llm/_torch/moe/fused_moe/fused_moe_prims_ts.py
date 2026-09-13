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
"""Vendored FlashInfer PrimsTS MoE backends for Blackwell."""

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.utils import ActivationType, ActType_TrtllmGen
from tensorrt_llm.models.modeling_utils import QuantAlgo

from .activation import ActivationParamShape, MoEActivationSupport
from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from .impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoEStaticCapability,
)
from .impl_environment import MoEDep
from .impl_identity import MoEImplDescriptor, MoEImplId, register_moe_impl
from .interface import MoESchedulerKind, _reject
from .moe_op_backend import TRTLLMOpBackend
from .routing import DeepSeekV3MoeRoutingMethod


def _supports_kimi_routing(routing_method, num_experts, hidden_size):
    return (
        isinstance(routing_method, DeepSeekV3MoeRoutingMethod)
        and routing_method.routing_impl.is_fused
        and num_experts == 896
        and hidden_size == 3584
        and routing_method.top_k == 16
        and routing_method.n_group == 1
        and routing_method.topk_group == 1
    )


class PrimsTSOpBackend(TRTLLMOpBackend):
    """Use native quantization and PrimTS grouped GEMMs."""

    def __init__(self, routing_method=None):
        self._routing_method = routing_method

    def run_fp4_block_scale_moe(
        self,
        router_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize=True,
        topk_weights=None,
        topk_ids=None,
        valid_hidden_size=None,
        valid_intermediate_size=None,
        enable_pdl=None,
        gated_act_type=0,
        output=None,
        tune_max_num_tokens=8192,
        use_dp=False,
    ):
        from ..custom_ops.prims_ts_moe import prims_ts_moe
        from ..flashinfer.tllm_enums import ActivationType as PrimsActivation

        if (topk_ids is None) != (topk_weights is None):
            raise ValueError(
                "PrimsTS requires both top-k indices and weights when routing is precomputed."
            )
        if gemm1_bias is not None or gemm2_bias is not None:
            raise ValueError("PrimsTS expert bias is not supported.")
        if routing_bias is None and self._routing_method is not None:
            # Separated routing omits the bias from the native MoE call, but
            # its expert-popularity distribution still matters for profiling.
            routing_bias = getattr(self._routing_method, "e_score_correction_bias", None)
        activations = {
            int(ActType_TrtllmGen.SwiGlu): int(PrimsActivation.Swiglu),
            int(ActType_TrtllmGen.SiTu): int(PrimsActivation.Situ),
        }
        if gated_act_type not in activations:
            raise ValueError(f"Unsupported PrimTS activation: {gated_act_type}")
        hidden_size = gemm1_weights.shape[-1] * 2
        fused_logits = None
        if topk_ids is None:
            if router_logits is None or self._routing_method is None:
                raise ValueError(
                    "PrimsTS requires routing logits and a routing method, or precomputed top-k routing."
                )
            # Keep a single logits signature across K3 token buckets. The
            # runner selects routing per profile, so warming up one token also
            # tunes the separated path needed by medium decode batches.
            if (
                _supports_kimi_routing(self._routing_method, num_experts, hidden_size)
                and hidden_states.dtype in (torch.uint8, torch.float8_e4m3fn)
                and router_logits.dtype == torch.float32
                and routing_bias is not None
                and routing_bias.dtype == torch.float32
            ):
                fused_logits = router_logits
            else:
                topk_ids, topk_weights = self._routing_method.apply(router_logits)
                topk_weights = topk_weights.to(torch.bfloat16)
        if gemm2_weights.shape[-2] != hidden_size:
            raise ValueError("PrimsTS requires equal padded FC1 input and FC2 output widths.")
        if output is None:
            output = torch.empty(
                (hidden_states.shape[0], hidden_size),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
        elif output.shape != (hidden_states.shape[0], hidden_size):
            raise ValueError("PrimsTS output shape must match the quantized hidden states.")
        permuted, expanded_map, fused_weights = prims_ts_moe(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            topk_ids,
            topk_weights,
            num_experts,
            local_expert_offset,
            activations[gated_act_type],
            do_finalize,
            output,
            enable_pdl=get_env_enable_pdl() if enable_pdl is None else enable_pdl,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp,
            routing_method_type=int(routing_method_type),
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            routing_bias=routing_bias,
            router_logits=fused_logits,
            top_k=top_k,
            use_hybrid_routing=fused_logits is not None,
        )
        if do_finalize:
            return output
        return permuted, topk_weights if topk_weights is not None else fused_weights, expanded_map


class PrimsTSFusedMoE(TRTLLMGenFusedMoE):
    """Share the TRTLLM FP4 weight ABI and external communication scheduler."""

    _SUPPORTED_QUANT_ALGOS = {QuantAlgo.NVFP4, QuantAlgo.W4A8_MXFP4_MXFP8}
    capabilities = MoEStaticCapability()
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.SiTu}),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        if d.eplb_enabled:
            return _reject(MoERejectReason.EPLB_UNSUPPORTED, "PrimsTS does not support EPLB.")
        if d.moe_lora_enabled:
            return _reject(MoERejectReason.LORA_UNSUPPORTED, "PrimsTS does not support MoE LoRA.")
        if d.env.sm not in (100, 103):
            return _reject(MoERejectReason.SM_UNSUPPORTED, "PrimsTS requires SM100 or SM103.")
        if p.quant_algo not in cls._SUPPORTED_QUANT_ALGOS:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED, f"{cls.__name__} cannot serve {p.quant}."
            )
        if p.activation_type not in cls.activation_support.kinds or p.bias or p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "PrimsTS supports bias-free SwiGLU and SiTU.",
            )
        # The shared MXFP4 loader pads FC1's input to 512, while its FC2
        # output uses 128. The adapter requires the two physical widths to
        # agree. NVFP4's padded method similarly rounds large inputs to 256.
        hidden_alignment = 128
        if p.quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
            hidden_alignment = 512
        elif p.hidden_size is not None and p.hidden_size > 1024:
            hidden_alignment = 256
        for size, alignment in (
            (p.hidden_size, hidden_alignment),
            (p.intermediate_size, 128 * d.tp_size),
        ):
            if size is not None and size % alignment:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED,
                    f"PrimsTS requires dimension {size} aligned to {alignment} with this weight layout.",
                )
        if not d.env.has_dep(MoEDep.PRIMS_TS):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "PrimsTS requires CUTLASS DSL primitives and task scheduling.",
            )
        return super().can_implement(p, d)

    def _select_op_provider(self) -> None:
        self.use_flashinfer = False
        self.op_backend = PrimsTSOpBackend(self.routing_method)

    def _requires_separated_routing(self) -> bool:
        return not (
            (self.has_nvfp4 or self.has_w4a8_mxfp4_mxfp8)
            and self.activation_type == ActivationType.SiTu
            and _supports_kimi_routing(self.routing_method, self.num_experts, self.hidden_size)
        )

    def _resolve_fused_shared_expert(self) -> None:
        self.num_fused_shared_expert = 0


@register_moe_impl
class PrimsTSNvfp4FusedMoE(PrimsTSFusedMoE):
    """NVFP4 weights and activations with fused SwiGLU or SiTU."""

    _SUPPORTED_QUANT_ALGOS = {QuantAlgo.NVFP4}
    descriptor = MoEImplDescriptor(
        identity=MoEImplId("flashinfer", "cutedsl", "prims_ts", "nvfp4"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=PrimsTSFusedMoE.capabilities,
        input_requirement=PrimsTSFusedMoE.input_requirement,
        doc="PrimsTS NVFP4 grouped GEMM on Blackwell.",
    )
    scheduler_kind = descriptor.scheduler_kind


@register_moe_impl
class PrimsTSMxfp4Mxfp8FusedMoE(PrimsTSFusedMoE):
    """MXFP4 weights and MXFP8 activations with fused SwiGLU or SiTU."""

    _SUPPORTED_QUANT_ALGOS = {QuantAlgo.W4A8_MXFP4_MXFP8}
    descriptor = MoEImplDescriptor(
        identity=MoEImplId("flashinfer", "cutedsl", "prims_ts", "w4a8_mxfp4_mxfp8"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=PrimsTSFusedMoE.capabilities,
        input_requirement=PrimsTSFusedMoE.input_requirement,
        doc="PrimsTS MXFP4 x MXFP8 grouped GEMM on Blackwell.",
    )
    scheduler_kind = descriptor.scheduler_kind
