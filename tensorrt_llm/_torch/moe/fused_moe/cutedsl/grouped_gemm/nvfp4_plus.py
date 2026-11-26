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
"""``TrtllmCutedslGroupedGemmNvfp4PlusImpl``: ``trtllm.cutedsl.grouped_gemm.nvfp4_plus``."""

import torch

from tensorrt_llm.quantization.utils.fp4_utils import NVFP4_PLUS_SF_VEC_SIZE

from .....autotuner import AutoTuner
from .....utils import ActivationType, EventType, Fp4QuantizedTensor
from ...activation import ActivationParamShape, MoEActivationSupport
from ...cutlass.eligibility import SmSupport
from ...impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    nvfp4_fc1_row_alignment_rejection,
    require_comm_plan,
)
from ...impl_identity import MoEImplDescriptor, register_moe_impl
from ...interface import MoESchedulerKind
from ...quantization import NVFP4PlusCuteDslFusedMoEMethod
from ..identity import grouped_gemm_identity
from ..kernel_inputs import NvFp4WeightView, finalize_routing_scales
from ..runners import CuteDslFusedMoENvfp4Runner
from .base import CuteDslFusedMoEBase
from .eligibility import check_grouped_gemm_leaf

# The FC1 kernel gathers a token's UE5M3 scales 16 bytes (two K tiles) at a
# time, so a token's scale row must be a whole number of 16-byte copies.
_HIDDEN_SIZE_ALIGNMENT = 16 * NVFP4_PLUS_SF_VEC_SIZE


def _hidden_size_alignment_rejection(p: MoEProblem) -> MoEEligibility | None:
    """Reject a hidden size whose UE5M3 scale row is not 16-byte aligned."""
    if p.hidden_size is None or p.hidden_size % _HIDDEN_SIZE_ALIGNMENT == 0:
        return None
    return MoEEligibility.no(
        MoERejectReason.SHAPE_UNALIGNED,
        f"NVFP4_PLUS MoE requires hidden_size to be a multiple of "
        f"{_HIDDEN_SIZE_ALIGNMENT}, but got {p.hidden_size}.",
    )


@register_moe_impl
class TrtllmCutedslGroupedGemmNvfp4PlusImpl(CuteDslFusedMoEBase):
    """``trtllm.cutedsl.grouped_gemm.nvfp4_plus``: CuteDSL NVFP4_PLUS grouped GEMM.

    NVFP4_PLUS keeps NVFP4's E2M1 values but stores one UE5M3 scale per 32
    values and no global scale, so MXFP4 checkpoints load without
    requantization. It runs the Rubin NVFP4 grouped-GEMM kernels with UE5M3
    scales: gather + FC1 + SwiGLU (+ FP4 quantization of the FC2 input), then
    FC2 + finalize. SM107 only.
    """

    descriptor = MoEImplDescriptor(
        identity=grouped_gemm_identity("nvfp4_plus"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        # EPLB reuses the NVFP4 CuteDSL weight pipeline: the host mirrors of
        # migrated experts take the same E8M0 -> UE5M3 load hooks and the same
        # gate16/up16 + swizzle preparation as the device weights.
        capabilities=MoEStaticCapability(
            supports_eplb=True,
            supports_apply_router_weight_on_input=True,
        ),
        input_requirement=MoEInputRequirement(routing_scales_dtype=torch.float32),
        doc="CuTe DSL NVFP4_PLUS (E2M1 with UE5M3 scales per 32 values) grouped GEMM: "
        "gather + FC1 + fused SwiGLU, then FC2 + finalize; SM107.",
    )
    sm_support = SmSupport(allowed=frozenset({107}))

    # The clamp is a kernel-cache-key scalar and the epilogue has no "clamp
    # absent" branch, so an absent clamp is +inf, not None.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu}),
        limit=ActivationParamShape.UNIFORM_SCALAR,
        limit_when_absent=float("inf"),
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        # process_weights_after_loading() unswizzles the FC1 block scales,
        # and FC2 reads its input scales in whole 128x4 tiles.
        return check_grouped_gemm_leaf(
            cls,
            p,
            d,
            nvfp4_fc1_row_alignment_rejection(p, d),
            _hidden_size_alignment_rejection(p),
        )

    def _get_quant_method(self) -> object:
        return NVFP4PlusCuteDslFusedMoEMethod()

    def quantize_input(
        self,
        x: torch.Tensor,
        post_quant_comm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``x`` [num_tokens, hidden_size] to NVFP4_PLUS.

        Returns the packed E2M1 values [num_tokens, hidden_size // 2] and the
        row-major UE5M3 scales [num_tokens, hidden_size // 32], which the FC1
        kernel gathers by token and alltoall/allgather move per token.
        """
        del post_quant_comm  # Both paths produce the same layout.
        if isinstance(x, Fp4QuantizedTensor):
            raise TypeError("NVFP4_PLUS MoE quantizes its own BF16 input")
        # The kernel module needs the CuTe DSL; it is only imported on SM107.
        from .....kernels.rubin.nvfp4_plus_quantize import nvfp4_plus_quantize

        return nvfp4_plus_quantize(x, sf_swizzled=False)

    def _build_local_weight_view(self) -> NvFp4WeightView:
        """Build the weight view from this backend's per-layer weights."""
        return NvFp4WeightView(
            w3_w1_weight=self.w3_w1_weight,
            fc1_weight_scale=self.quant_scales.fc1_weight_block,
            fc1_global_scale=self.quant_scales.fc1_global,
            w2_weight=self.w2_weight,
            fc2_weight_scale=self.quant_scales.fc2_weight_block,
            fc2_global_scale=self.quant_scales.fc2_global,
            expert_size_per_partition=self.expert_size_per_partition,
            slot_start=self.slot_start,
        )

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        """Run the NVFP4_PLUS MoE; returns final_hidden_states."""
        del workspace  # CuteDSL kernels allocate their own intermediates.
        plan = require_comm_plan(self, ctx)
        if ctx.x_ready_event is not None:
            ctx.x_ready_event.wait()
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = finalize_routing_scales(self, ctx)
        output_dtype = torch.bfloat16
        moe_output = plan.moe_output
        if moe_output is None:
            moe_output = torch.empty(
                (token_final_scales.size(0), self.hidden_size),
                dtype=output_dtype,
                device=ctx.x.device,
            )
        else:
            assert moe_output.size() == (token_final_scales.size(0), self.hidden_size)
            assert moe_output.dtype == output_dtype
        # Empty micro-batches are valid at the backend boundary. Avoid
        # entering autotuning because its synthetic grouped-GEMM inputs
        # require at least one output row.
        if token_selected_experts.size(0) == 0:
            return moe_output

        weight_view = self._build_local_weight_view()
        runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=self.run_moe_nvfp4_plus_impl,
            num_experts=self.num_slots,
            top_k=token_selected_experts.size(-1),
            num_local_experts=weight_view.expert_size_per_partition,
            local_expert_offset=weight_view.slot_start,
            enable_finalize_fusion=self.use_fused_finalize,
            enable_alltoall=plan.enable_alltoall,
            scaling_vector_size=self.scaling_vector_size,
        )
        inputs = [
            ctx.x,
            token_selected_experts,
            token_final_scales,
            ctx.x_sf,
            moe_output,
            weight_view,
        ]
        _, best_tactic = AutoTuner.get().choose_one(
            "CuteDslFusedMoE::run_moe_nvfp4_plus",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def run_moe_nvfp4_plus_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weight_view: NvFp4WeightView,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        """NVFP4_PLUS MoE through the Rubin gather/act-fusion and finalize ops."""
        effective_top_k = token_selected_experts.size(1)
        esp = weight_view.expert_size_per_partition
        slot_start = weight_view.slot_start
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            _total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            local_expert_offset=slot_start,
            local_num_experts=esp,
            tile_tokens_dim=tile_size,
        )

        def memset_moe_output() -> None:
            torch.ops.trtllm.moe_output_memset_inplace(
                input=moe_output,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                tile_tokens_dim=tile_size,
                top_k=effective_top_k,
                ep_size=self.mapping.moe_ep_size,
                enable_alltoall=enable_alltoall,
            )

        # The finalize kernel scatter-adds into moe_output, so clear the rows it
        # touches; overlap that with FC1 when an aux stream is available.
        has_aux_streams = self._has_moe_output_memset_aux_stream()
        if has_aux_streams:
            memset_stream = self._moe_output_memset_run_stream()
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(memset_stream)
            with torch.cuda.stream(memset_stream):
                self.event_dict[EventType.Main].wait()
                memset_moe_output()
                self.event_dict[EventType.MoeOutputMemset].record()

        # FC1: gather + grouped GEMM + SwiGLU + NVFP4_PLUS quantization of the
        # FC2 input. NVFP4_PLUS has no global scale, so the per-expert alpha
        # and the FC2 input scale the kernels take are the ones the weight
        # loader leaves in place.
        x, x_sf = torch.ops.trtllm.cute_dsl_nvfp4_plus_gather_grouped_gemm_act_fusion_rubin(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=weight_view.w3_w1_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale=weight_view.fc1_weight_scale.view(torch.uint8),
            alpha=weight_view.fc1_global_scale,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=self.fc2_input_scale,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
            activation_type=int(self.activation_type),
            swiglu_limit_scalar=self.act_clamp,
        )

        if has_aux_streams:
            self.event_dict[EventType.MoeOutputMemset].wait()
        else:
            memset_moe_output()

        # FC2: grouped GEMM + finalize (scatter-add with the routing scales).
        torch.ops.trtllm.cute_dsl_nvfp4_plus_grouped_gemm_finalize_inplace_rubin(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=weight_view.w2_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
            alpha=weight_view.fc2_global_scale,
            output=moe_output,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
            output_dtype=moe_output.dtype,
        )
        return moe_output
