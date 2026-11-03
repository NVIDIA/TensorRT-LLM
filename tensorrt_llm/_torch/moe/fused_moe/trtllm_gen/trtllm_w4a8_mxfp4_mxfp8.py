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

import functools
import os
from typing import Sequence, Tuple

import torch

from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
    pin_mxe4m3_mxe2m1_moe_row_tactics,
    pin_mxe4m3_mxe2m1_moe_tactics,
)
from tensorrt_llm._torch.utils import ActType_TrtllmGen, MxFp8QuantizedTensor
from tensorrt_llm._utils import is_sm_100f

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..routing import DeepSeekV3MoeRoutingMethod
from .eligibility import check_trtllm_gen_leaf
from .fp4_block_scale import TRTLLMGenW4a8Mxfp4Mxfp8Base
from .identity import PROVIDER_TRTLLM, trtllm_gen_descriptor
from .kernel_inputs import to_trtllm_gen_act_type


@functools.cache
def _runner_has_unfinalized() -> bool:
    runner = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(0, True)
    return hasattr(runner, "run_moe_unfinalized")


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

    def pin_decode_tactics(
        self,
        pins: Sequence[Tuple[int, int, int, int]],
        row_tactics: Sequence[Tuple[int, int, int, int]] = (),
    ) -> None:
        """Offer the autotuner only tactic ``[tile_n, config]`` for tuned token counts in ``[lo, hi]``.

        ``pins`` holds ``(lo, hi, tile_n, config)`` entries. They apply to every
        runner call with this layer's local intermediate size, hidden size and
        activation; a pinned tactic the runner does not list for a shape is
        ignored there. ``row_tactics`` (same layout, over the real token count
        of a call) replace the bucket's choice for those row counts; only
        tactics bitwise-equal to it belong there. Takes effect once the expert
        weights exist (now, or when the loader binds them).
        """
        self._decode_tactic_pins = tuple(tuple(pin) for pin in pins)
        self._decode_row_tactics = tuple(tuple(row) for row in row_tactics)
        self._register_decode_tactic_pins()

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        self._register_decode_tactic_pins()

    def _register_decode_tactic_pins(self) -> None:
        pins = getattr(self, "_decode_tactic_pins", ())
        weight = getattr(self, "w3_w1_weight", None)
        if not pins or weight is None:
            return
        act_type = to_trtllm_gen_act_type(self.activation_type)
        gated = act_type not in (int(ActType_TrtllmGen.Relu2), int(ActType_TrtllmGen.Silu))
        pin_mxe4m3_mxe2m1_moe_tactics(
            intermediate_size=weight.shape[-2] // (2 if gated else 1),
            hidden_size=self.hidden_size,
            act_type=act_type,
            pins=pins,
        )
        rows = getattr(self, "_decode_row_tactics", ())
        if rows:
            pin_mxe4m3_mxe2m1_moe_row_tactics(
                intermediate_size=weight.shape[-2] // (2 if gated else 1),
                hidden_size=self.hidden_size,
                act_type=act_type,
                rows=rows,
            )

    def supports_unfinalized_output(self) -> bool:
        """The native runner returns the unfinalized triple when the build has it.

        ``trtllm::mxe4m3_mxe2m1_block_scale_moe_runner_unfinalized`` needs the
        runner's ``run_moe_unfinalized`` method; builds that predate it can only
        finalize.
        """
        return _runner_has_unfinalized()

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def try_fused_route_quant(
        self,
        x: torch.Tensor | MxFp8QuantizedTensor,
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
        if os.environ.get("TLLM_K3_DISABLE_FUSED_ROUTE_QUANT", "0") == "1" or isinstance(
            x, MxFp8QuantizedTensor
        ):
            return None
        # A top-k precomputed together with the runner's routing (Kimi K3.1's routing tail,
        # ``presorted``) replaces the fused op's routing.
        if getattr(self.routing_method, "presorted", None) is not None:
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
