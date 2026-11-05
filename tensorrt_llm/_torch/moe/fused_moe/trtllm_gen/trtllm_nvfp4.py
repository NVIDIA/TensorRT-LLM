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
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

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
    from ....moe.kernels.blackwell.dsv4_route_quant import (
        ROUTING_TABLE_TILES,
        can_build_routing_tables,
        dsv4_route,
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

    #: The native op's pre-routed entry point (``run_moe_prerouted``).
    supports_prerouted_routing = True

    #: Whether routing runs as one launch, fused with quantization, and with routing tables; set
    #: after loading.
    _fuses_route = False
    _fuses_route_quant = False
    _builds_routing_tables = False

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def cache_derived_state(self) -> None:
        """Also decide, once loaded, whether routing fuses with the input quantization."""
        super().cache_derived_state()
        self._fuses_route = (
            IS_CUTLASS_DSL_AVAILABLE and self.routing_method.sqrt_softplus_bias_selection
        )
        self._fuses_route_quant = (
            self._fuses_route
            and self.dtype == torch.bfloat16
            and self.fc31_act_scale is None
            and self.w3_w1_weight.shape[-1] * 2 == self.hidden_size
        )
        # Tables number the rank's experts, not slots, and attention-DP dispatch would regroup
        # tokens.
        self._builds_routing_tables = (
            self._fuses_route
            and not self.use_dp
            and self.layer_load_balancer is None
            and self.num_slots == self.num_experts
        )
        self._table_tile_by_rows = {}

    def ran_on_routing_tables(self, rows: int, tile: int | None) -> None:
        if tile is None:
            self._table_tile_by_rows.pop(rows, None)
        else:
            self._table_tile_by_rows[rows] = tile

    def _routing_table_tiles(self, rows: int) -> tuple[int, ...]:
        """The table of the tile a run of ``rows`` tokens used once one ran on tables, else every
        tile's (a GEMM tactic of another tile routes in the runner and resets it)."""
        tile = self._table_tile_by_rows.get(rows)
        return ROUTING_TABLE_TILES if tile is None else (tile,)

    def _routing_table_experts(self, rows: int) -> tuple[int, int] | None:
        """The (first, count) experts this rank's routing tables cover at ``rows`` tokens, if any."""
        if self._builds_routing_tables and can_build_routing_tables(
            rows, self.expert_size_per_partition, self.num_experts
        ):
            return self.slot_start, self.expert_size_per_partition
        return None

    def try_fused_route(
        self, x: Fp4QuantizedTensor, router_logits: torch.Tensor
    ) -> MoEFusedRouteQuant | None:
        """Sqrt-softplus routing in one launch, with the runner's routing tables for small batches,
        for an input its producer already quantized to NVFP4 (e.g. the model's fused boundary)."""
        # A pre-quant scale (AWQ/ARC) must be applied before the producer's quantization.
        if not self._fuses_route or x.is_sf_swizzled or self.fc31_act_scale is not None:
            return None
        rows = x.shape[0]
        local_experts = self._routing_table_experts(rows)
        with_tables = local_experts is not None
        tiles = self._routing_table_tiles(rows)
        ids, weights, tables = dsv4_route(
            router_logits,
            self.routing_method.e_score_correction_bias,
            self.routing_method.top_k,
            self.routing_method.routed_scaling_factor,
            self.input_requirement.routing_scales_dtype,
            self.hidden_size,
            with_routing_tables=with_tables,
            local_experts=local_experts,
            routing_tiles=tiles,
        )
        return MoEFusedRouteQuant(
            ids,
            weights,
            x.fp4_tensor,
            x.scaling_factor.view(rows, -1),
            routing_tables=MoERoutingTables(tables, tiles) if with_tables else None,
        )

    def try_fused_route_quant(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> MoEFusedRouteQuant | None:
        """Sqrt-softplus routing and NVFP4 input quantization in one launch, with the runner's
        routing tables for small batches."""
        if not self._fuses_route_quant:
            return None
        local_experts = self._routing_table_experts(x.shape[0])
        with_tables = local_experts is not None
        tiles = self._routing_table_tiles(x.shape[0])
        ids, weights, x_fp4, x_sf, tables = dsv4_route_nvfp4_quant(
            router_logits,
            self.routing_method.e_score_correction_bias,
            x,
            self.fc31_input_scale,
            self.routing_method.top_k,
            self.routing_method.routed_scaling_factor,
            self.input_requirement.routing_scales_dtype,
            with_routing_tables=with_tables,
            local_experts=local_experts,
            routing_tiles=tiles,
        )
        return MoEFusedRouteQuant(
            ids,
            weights,
            x_fp4,
            x_sf,
            routing_tables=MoERoutingTables(tables, tiles) if with_tables else None,
        )
