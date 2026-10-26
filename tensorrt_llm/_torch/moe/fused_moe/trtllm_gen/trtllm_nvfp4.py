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
"""``trtllm.trtllm_gen.fused_moe.nvfp4``."""

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEFusedRouteQuant,
    MoEProblem,
    MoERoutingTables,
)
from ..impl_identity import register_moe_impl
from .eligibility import check_trtllm_gen_leaf
from .fp4_block_scale import TRTLLMGenNvfp4Base
from .identity import PROVIDER_TRTLLM, trtllm_gen_descriptor

if IS_CUTLASS_DSL_AVAILABLE:
    from ....cute_dsl_kernels.blackwell.dsv4_route_quant import (
        ROUTING_TABLE_TILES,
        can_build_routing_tables,
        dsv4_route_nvfp4_quant,
    )


@register_moe_impl
class TrtllmTrtllmGenNvfp4Impl(TRTLLMGenNvfp4Base):
    """``trtllm.trtllm_gen.fused_moe.nvfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "nvfp4", "TRTLLM-Gen batched-GEMM cubins over NVFP4, SM100 family."
    )

    #: The group-16 fused SiTu cubins exist for this format, but only the
    #: native op backend calls them, so eligibility is per leaf and not on
    #: ``TRTLLMGenNvfp4Base`` (which the FlashInfer leaf shares).
    supports_situ = True

    #: Whether routing fuses with quantization, and with routing tables; set after loading.
    _fuses_route_quant = False
    _builds_routing_tables = False

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def cache_derived_state(self) -> None:
        """Also decide, once loaded, whether routing fuses with the input quantization."""
        super().cache_derived_state()
        self._fuses_route_quant = (
            IS_CUTLASS_DSL_AVAILABLE
            and self.routing_method.sqrt_softplus_bias_selection
            and self.dtype == torch.bfloat16
            and self.fc31_act_scale is None
            and self.w3_w1_weight.shape[-1] * 2 == self.hidden_size
        )
        # Tables number experts, not slots, and attention-DP dispatch would regroup tokens.
        self._builds_routing_tables = (
            self._fuses_route_quant
            and not self.use_dp
            and self.expert_size_per_partition == self.num_slots == self.num_experts
        )

    def try_fused_route_quant(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> MoEFusedRouteQuant | None:
        """Sqrt-softplus routing and NVFP4 input quantization in one launch, with the runner's
        routing tables for small batches."""
        if not self._fuses_route_quant:
            return None
        with_tables = self._builds_routing_tables and can_build_routing_tables(
            x.shape[0], self.num_experts
        )
        ids, weights, x_fp4, x_sf, tables = dsv4_route_nvfp4_quant(
            router_logits,
            self.routing_method.e_score_correction_bias,
            x,
            self.fc31_input_scale,
            self.routing_method.top_k,
            self.routing_method.routed_scaling_factor,
            self.input_requirement.routing_scales_dtype,
            with_routing_tables=with_tables,
        )
        return MoEFusedRouteQuant(
            ids,
            weights,
            x_fp4,
            x_sf,
            routing_tables=MoERoutingTables(tables, ROUTING_TABLE_TILES) if with_tables else None,
        )
