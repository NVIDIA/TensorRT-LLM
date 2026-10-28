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
"""``trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8``."""

import os

import torch

from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor, MxFp8QuantizedTensor
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext
from ..impl_identity import register_moe_impl
from ..interface import MoEWeightLoadingMode
from ..quantization import MXFP4AsNVFP4TRTLLMGenFusedMoEMethod
from ..routing import DeepSeekV3MoeRoutingMethod
from .eligibility import check_trtllm_gen_leaf
from .fp4_block_scale import TRTLLMGenNvfp4Base, TRTLLMGenW4a8Mxfp4Mxfp8Base
from .identity import PROVIDER_TRTLLM, trtllm_gen_descriptor
from .trtllm_nvfp4 import TrtllmTrtllmGenNvfp4Impl

#: Opt-in (default off): run this leaf's MXFP4 experts as NVFP4 x NVFP4 (W4A4).
#: The stored MXFP4 checkpoint is repacked at load time (FP4 codes unchanged,
#: each UE8M0 group-32 scale written as two E4M3 group-16 scales under a
#: per-expert power-of-two global scale), and the TRTLLM-Gen NVFP4 runner with
#: NVFP4 activations replaces the MXFP8-activation runner. Meant for role-scoped
#: use (e.g. the prefill workers only); see ``mxfp4_as_nvfp4``.
_MXFP4_MOE_NVFP4_ACT_ENV = "TLLM_MXFP4_MOE_NVFP4_ACT"


@register_moe_impl
class TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl(TRTLLMGenW4a8Mxfp4Mxfp8Base):
    """``trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM,
        "w4a8_mxfp4_mxfp8",
        "TRTLLM-Gen batched-GEMM cubins over MXFP4 weights with MXFP8 activations.",
    )

    #: The group-32 fused SiTu cubins; see ``TrtllmTrtllmGenNvfp4Impl`` on why
    #: this is declared per leaf rather than on the shared format base.
    supports_situ = True

    # ---- opt-in NVFP4 (W4A4) execution of the MXFP4 experts ----------------
    @property
    def mxfp4_as_nvfp4(self) -> bool:
        """Whether this layer runs its MXFP4 experts on the NVFP4 W4A4 runner.

        Settled once, at the first ask (``create_weights``), from the opt-in
        env var and the conditions the repack needs: the gpt-oss SwiGLU with a
        clamp (it bounds the FC2 input, which sets the static FC2 activation
        scale), fused gate/up checkpoint tensors (the loader that repacks
        them), and no MoE TP (the repack shards nothing). The model loader reads
        it to hand the layer NVFP4-format tensors.
        """
        cached = self.__dict__.get("_mxfp4_as_nvfp4")
        if cached is not None:
            return cached
        enabled = os.environ.get(_MXFP4_MOE_NVFP4_ACT_ENV, "0") == "1"
        if enabled:
            clamp = getattr(self.activation, "clamp", None)
            reasons = [
                why
                for bad, why in (
                    (self.activation_type != ActivationType.SwigluBias, "activation is not SwigluBias"),
                    (clamp is None, "SwigluBias has no clamp"),
                    (
                        self.weight_loading_mode != MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
                        "weights are not FUSED_GATE_UP_PROJ",
                    ),
                    (self.mapping.moe_tp_size != 1, f"moe_tp_size={self.mapping.moe_tp_size}"),
                )
                if bad
            ]
            if reasons:
                logger.warning_once(
                    f"{_MXFP4_MOE_NVFP4_ACT_ENV}=1 ignored: {'; '.join(reasons)}",
                    key="mxfp4_moe_nvfp4_act_ignored",
                )
                enabled = False
            else:
                logger.info_once(
                    f"{_MXFP4_MOE_NVFP4_ACT_ENV}=1: MXFP4 experts repacked to NVFP4 at load, "
                    "NVFP4 activations (W4A4) on the TRTLLM-Gen NVFP4 runner",
                    key="mxfp4_moe_nvfp4_act_engaged",
                )
        self.__dict__["_mxfp4_as_nvfp4"] = enabled
        return enabled

    def _get_quant_method(self) -> object:
        if self.mxfp4_as_nvfp4:
            return MXFP4AsNVFP4TRTLLMGenFusedMoEMethod()
        return super()._get_quant_method()

    def quantize_input(
        self,
        x: torch.Tensor | Fp4QuantizedTensor | MxFp8QuantizedTensor,
        post_quant_comm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mxfp4_as_nvfp4:
            return TRTLLMGenNvfp4Base.quantize_input(self, x, post_quant_comm)
        return super().quantize_input(x, post_quant_comm)

    def get_nvfp4_input_quant_params(self) -> tuple[torch.Tensor, int] | None:
        if self.mxfp4_as_nvfp4:
            return TRTLLMGenNvfp4Base.get_nvfp4_input_quant_params(self)
        return super().get_nvfp4_input_quant_params()

    def _get_direct_moe_output_hidden_size(self, ctx: MoERunContext) -> int | None:
        if self.mxfp4_as_nvfp4:
            return TrtllmTrtllmGenNvfp4Impl._get_direct_moe_output_hidden_size(self, ctx)
        return super()._get_direct_moe_output_hidden_size(ctx)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def try_fused_route_quant(
        self,
        x: torch.Tensor | Fp4QuantizedTensor | MxFp8QuantizedTensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Fuse Kimi K3 no-aux routing and MXFP8 input quantization.

        The op is MXFP8-activation and native-cubin only, which is exactly this
        identity, so no format or provider test is needed here.

        Beyond that the op is specialized to the K3 decode shape: it hardcodes
        896 experts, top-16, hidden 3584 and at most 64 tokens. The checks
        below mirror its ``TORCH_CHECK``s so a miss declines quietly instead of
        raising, which keeps every other model and shape on the unfused path.
        """
        if (
            os.environ.get("TLLM_K3_DISABLE_FUSED_ROUTE_QUANT", "0") == "1"
            or isinstance(x, (Fp4QuantizedTensor, MxFp8QuantizedTensor))
            or self.mxfp4_as_nvfp4
        ):
            return None

        # Direct construction reaches here without a resolution query, so the
        # architecture the cubins need is still worth asserting.
        if not is_sm_100f() or not isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod):
            return None

        routing = self.routing_method.routing_impl
        bias = self.routing_method.e_score_correction_bias
        if (
            not routing.is_fused
            or routing.n_group != 1
            or routing.topk_group != 1
            or routing.top_k != 16
            or router_logits.ndim != 2
            or router_logits.shape[1] != 896
            or router_logits.dtype != torch.float32
            or not router_logits.is_contiguous()
            or bias.dtype != torch.float32
            or not bias.is_contiguous()
            or x.ndim != 2
            or x.shape != (router_logits.shape[0], 3584)
            or not 0 < x.shape[0] <= 64
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            return None

        return torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
            router_logits, bias, x, routing.routed_scaling_factor
        )
