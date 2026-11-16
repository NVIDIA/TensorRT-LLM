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

from tensorrt_llm._torch.utils import MxFp8QuantizedTensor
from tensorrt_llm._utils import is_sm_100f

from ..impl_contract import MoEDeployment, MoEEligibility, MoEFusedRouteQuant, MoEProblem
from ..impl_identity import register_moe_impl
from ..routing import DeepSeekV3MoeRoutingMethod, DeepSeekV4MoeRoutingMethod
from .eligibility import check_trtllm_gen_leaf
from .fp4_block_scale import TRTLLMGenW4a8Mxfp4Mxfp8Base
from .identity import PROVIDER_TRTLLM, trtllm_gen_descriptor


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

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def try_fused_route_quant(
        self,
        x: torch.Tensor | MxFp8QuantizedTensor,
        router_logits: torch.Tensor,
    ) -> MoEFusedRouteQuant | None:
        """Fuse the routing and the MXFP8 input quantization: the DSpark draft bank's or Kimi K3's."""
        # Checked first: a pre-quantized input declines before any routing or SM probe.
        if isinstance(x, MxFp8QuantizedTensor):
            return None
        if isinstance(self.routing_method, DeepSeekV4MoeRoutingMethod):
            return self._try_dsv4_draft_route_quant(x, router_logits)
        return self._try_kimi_k3_route_quant(x, router_logits)

    def _try_dsv4_draft_route_quant(
        self,
        x: torch.Tensor | MxFp8QuantizedTensor,
        router_logits: torch.Tensor,
    ) -> MoEFusedRouteQuant | None:
        """The DeepSeek-V4 DSpark draft bank's routing (top-3 of 128, bias-selected) and the MXFP8
        quantization of its input in one launch: the ids and weights of
        ``DeepSeekV4MoeRoutingMethod.apply`` in the expert kernels' dtype and the bytes of
        ``quantize_input``. Declines for any other bank, a hashed gate, a routing method whose
        ``apply`` was replaced on the instance, or an input that needs padding.
        """
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        routing = self.routing_method
        if (
            not IS_CUTLASS_DSL_AVAILABLE
            or routing.is_hashed
            or "apply" in vars(routing)
            or routing.top_k != 3
            or not isinstance(x, torch.Tensor)
            or x.dtype != torch.bfloat16
            or x.dim() != 2
            or not x.is_contiguous()
            or x.shape[1] != self.hidden_size
            or x.shape[1] % self.quant_method.input_hidden_alignment
            or router_logits.shape != (x.shape[0], 128)
            or router_logits.dtype != torch.float32
            or not router_logits.is_contiguous()
        ):
            return None
        bias = routing.callable_e_score_correction_bias()
        if bias is None or bias.dtype != torch.float32 or not bias.is_contiguous():
            return None
        from ....attention.kernels.blackwell.dspark.draft_routing import dspark_draft_route_mxfp8

        return MoEFusedRouteQuant(
            *dspark_draft_route_mxfp8(
                router_logits,
                bias,
                x,
                routing.top_k,
                routing.routed_scaling_factor,
                self.input_requirement.routing_scales_dtype or torch.float32,
            )
        )

    def _try_kimi_k3_route_quant(
        self,
        x: torch.Tensor | MxFp8QuantizedTensor,
        router_logits: torch.Tensor,
    ) -> MoEFusedRouteQuant | None:
        """Fuse Kimi K3 no-aux routing and MXFP8 input quantization.

        The op is MXFP8-activation and native-cubin only, which is exactly this
        identity, so no format test is needed here.

        Beyond that the op is specialized to the K3 decode shape: it hardcodes
        896 experts, top-16, hidden 3584 and at most 64 tokens. The checks
        below mirror its ``TORCH_CHECK``s so a miss declines quietly instead of
        raising, which keeps every other model and shape on the unfused path.
        """
        if os.environ.get("TLLM_K3_DISABLE_FUSED_ROUTE_QUANT", "0") == "1":
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

        return MoEFusedRouteQuant(
            *torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
                router_logits, bias, x, routing.routed_scaling_factor
            )
        )
