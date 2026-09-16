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
"""CuteDslFc12FusedMoE: ``trtllm.cutedsl.fused_fc12.nvfp4``.

FC1+FC2-fused CuteDSL NVFP4 MoE backend for Rubin (SM107). One persistent kernel
(``trtllm::cute_dsl_nvfp4_fc12_fused_rubin``) does gather + FC1 GEMM + SwiGLU +
requant + FC2 GEMM + finalize (scatter-add into ``moe_output``), so the FC1->FC2
intermediate never round-trips through global memory.

A leaf in the ``MoEImplBase`` sense: it declares its ``descriptor``, registers
itself, and owns the four abstract methods. The NVFP4 weight layout, quant method
(``NVFP4CuteDslFusedMoEMethod``), weight view and input quantization are the same
ones ``CuteDslFusedMoE`` uses for its two-op NVFP4 path; they are restated here
(not inherited) so this class serves exactly one format and carries no bf16 /
fp8 / locality-domain entry points it cannot execute. The outer autotune runner
reuses ``CuteDslFusedMoENvfp4Runner`` with the tile set narrowed to what the
fused kernel supports. Locality domains (uGPU) are not enabled for this backend.
"""

from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...autotuner import AutoTuner
from ...model_config import ModelConfig
from ...utils import ActivationType, AuxStreamType, EventType, Fp4QuantizedTensor
from .activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from .fused_moe_cute_dsl import CuteDslFusedMoENvfp4Runner, NvFp4WeightView
from .impl_base import MoEImplBase, apply_moe_impl_construction_state
from .impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    require_comm_plan,
)
from .impl_environment import MoEDep
from .impl_identity import MoEImplDescriptor, MoEImplId, register_moe_impl
from .interface import MoESchedulerKind, _reject
from .quantization import MoEWeightLoadingMode, NVFP4CuteDslFusedMoEMethod
from .routing import BaseMoeRoutingMethod


class CuteDslFc12FusedMoENvfp4Runner(CuteDslFusedMoENvfp4Runner):
    """Outer autotune runner for the fused FC12 backend.

    Identical to the parent except the routing-tile candidate set: the fused
    kernel supports the 128-wide tile (1-CTA) and the 256-wide tile (2-CTA,
    cluster (2,1)), so restrict ``_tile_sizes`` to ``[128, 256]`` (the parent
    also offers 512, which the fused v1 kernel does not support). The inner
    fused runner derives mma_tiler_m == tile_size and cluster M == tile_size //
    128 from the selected tile, mirroring the CuteDSL grouped-GEMM runners.
    """

    @staticmethod
    def _tile_sizes():
        return [128, 256]


@register_moe_impl
class CuteDslFc12FusedMoE(MoEImplBase):
    """``trtllm.cutedsl.fused_fc12.nvfp4``: FC1+FC2-fused CuteDSL NVFP4 MoE (Rubin/SM107).

    Args mirror ``CuteDslFusedMoE``; see the module docstring for what is
    shared with it and why it is restated rather than inherited.
    """

    # ConfigurableMoE reads this off the backend type: the fused kernel takes
    # per-tile expert ranges, so EP need not divide the expert count.
    _supports_non_divisible_ep: bool = True

    descriptor = MoEImplDescriptor(
        identity=MoEImplId("trtllm", "cutedsl", "fused_fc12", "nvfp4"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=MoEStaticCapability(
            supports_dwdp=True,
            supports_eplb=True,
            supports_apply_router_weight_on_input=True,
        ),
        input_requirement=MoEInputRequirement(routing_scales_dtype=torch.float32),
        doc="Rubin (SM107) NVFP4 fused FC1 + SwiGLU + FC2 + finalize persistent "
        "CuTe DSL grouped GEMM.",
    )
    # Taken off the descriptor rather than restated: the scheduler reads these
    # three, the registry publishes the descriptor, and one literal keeps them
    # from drifting apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    # The fused FC12 kernel has no activation selector: FC1 weights are the
    # gate/up pair and the epilogue is SwiGLU (+ clamp). Relu2 is not gated and
    # is not supported, so the resolver turns such layers down here.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu}),
        limit=ActivationParamShape.UNIFORM_SCALAR,
        limit_when_absent=float("inf"),
    )

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """FC12 fused CuteDSL: NVFP4 on Rubin (SM107) only.

        Pure gate -- reads only ``p`` and ``d`` (no ``get_sm_version()``, no
        import probes, no ``os.environ``); the frozen environment lives in
        ``d.env``, including whether the installed CuTe DSL carries the Rubin
        helpers (``MoEDep.CUTEDSL_RUBIN``).
        """
        sm_version = d.env.sm

        # Output is hardcoded to bfloat16, so activation must match.
        if p.dtype_act != torch.bfloat16:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"CuteDslFc12FusedMoE only supports bfloat16 activation, got {p.dtype_act}",
            )

        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "CuteDslFc12FusedMoE does not support swiglu_gptoss_style",
            )

        if p.quant_algo != QuantAlgo.NVFP4:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"CuteDslFc12FusedMoE only supports NVFP4, got quant_algo={p.quant_algo}",
            )

        if sm_version != 107:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"CuteDslFc12FusedMoE targets Rubin (SM107), got SM{sm_version}",
            )

        # The fused FC12 kernel scatter-adds FC2 straight into moe_output;
        # there is no unfused FC2 variant, so a deployment that disabled
        # finalize fusion (moe_disable_finalize_fusion, or any LoRA) cannot
        # be served. Same guard CuteDslFusedMoE applies on SM107.
        if not d.fused_finalize_enabled:
            return _reject(
                MoERejectReason.FINALIZE_FUSION_REQUIRED,
                "CuteDslFc12FusedMoE only has a fused-finalize FC2",
            )

        # Read off the collected environment so an offline tuner on a GPU-less
        # host reaches the verdict a serving process would.
        if not d.env.has_dep(MoEDep.CUTEDSL_RUBIN):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "CuteDslFc12FusedMoE (SM107 NVFP4) requires CuTe DSL Rubin support",
            )

        return MoEEligibility.ok()

    # ------------------------------------------------------------------
    # Construction and weights. Same shape as CuteDslFusedMoE's NVFP4 path,
    # minus the bf16 / fp8 / locality-domain machinery this backend never runs.
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ):
        super().__init__(eplb=None)
        apply_moe_impl_construction_state(
            self,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
            activation=activation,
            init_load_balancer=init_load_balancer,
        )
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # can_implement already turned down deployments without fused finalize;
        # kept as the runner's enable_finalize_fusion flag for cache-key parity
        # with CuteDslFusedMoE.
        self.use_fused_finalize = (
            not model_config.moe_disable_finalize_fusion and model_config.lora_config is None
        )
        # The scheduler records/waits ``event_dict[EventType.Main]`` around the
        # aux stream in multi-stream mode (moe_scheduler.py), so the event has to
        # exist even though the fused op issues its own moe_output memset.
        if self.aux_stream_dict is None:
            self.aux_stream_dict = {}
        self.event_dict = {EventType.Main: torch.cuda.Event()}
        self.scaling_vector_size = 16
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _get_quant_method(self):
        if (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
            and self.quant_config.layer_quant_mode.has_nvfp4()
        ):
            # Same weight layout as CuteDslFusedMoE's NVFP4 path; the fused
            # kernel consumes w3_w1 / w2 and their block scales unchanged.
            return NVFP4CuteDslFusedMoEMethod()
        raise ValueError(f"CuteDslFc12FusedMoE only supports NVFP4, got {self.quant_config}")

    def _check_configs(self):
        assert self._weights_created
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

    def _supports_load_balancer(self) -> bool:
        return True

    def supports_moe_output_in_alltoall_workspace(self):
        # The fused op finalizes straight into moe_output, which may be the
        # all-to-all combine workspace.
        return True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        super().load_weights(weights, allow_partial_loading=allow_partial_loading)
        # DWDP registers the freshly loaded expert weights with its handle
        # collector; same hook CuteDslFusedMoE exposes (supports_dwdp=True).
        dwdp_handle_collector = getattr(self, "dwdp_handle_collector", None)
        if dwdp_handle_collector is not None:
            dwdp_handle_collector.register_weights(self)

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

    # ------------------------------------------------------------------
    # Input quantization and dispatch (NVFP4 only)
    # ------------------------------------------------------------------
    def quantize_input(
        self, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
    ):
        """NVFP4-quantize the input; returns (x, x_sf) with x_sf reshaped to 2D.

        The 2D shape ``[num_tokens, ceil(hidden / scaling_vector_size)]`` is
        what alltoall / allgather expect for the scaling factors.
        """
        del post_quant_comm  # single path: always quantized ahead of comm
        assert self.has_nvfp4
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, (
                "Fp4QuantizedTensor should not be swizzled before communication"
            )
            x_row = x.shape[0]
            x, x_sf = x.fp4_tensor, x.scaling_factor
        else:
            x_row = x.shape[0]
            x, x_sf = torch.ops.trtllm.fp4_quantize(
                x, self.fc31_input_scale, self.scaling_vector_size, False, False
            )
        # ``view(0, -1)`` is ambiguous for an empty micro-batch; spell the
        # scale width out so empty and non-empty inputs take the same path.
        scale_cols = (self.hidden_size + self.scaling_vector_size - 1) // self.scaling_vector_size
        return x, x_sf.view(x_row, scale_cols)

    def run_moe(self, ctx: MoERunContext, *, workspace: Optional[dict] = None) -> torch.Tensor:
        del workspace  # the fused kernel allocates its own intermediates
        plan = require_comm_plan(self, ctx)
        return self.run_moe_nvfp4(
            x=ctx.x,
            token_selected_experts=ctx.token_selected_experts,
            token_final_scales=ctx.token_final_scales,
            x_sf=ctx.x_sf,
            moe_output=plan.moe_output,
            enable_alltoall=plan.enable_alltoall,
            weight_view=self._build_local_weight_view(),
        )

    # ------------------------------------------------------------------
    # Fused-kernel dispatch. Mirrors the parent's non-uGPU NVFP4 path but
    # (1) uses a distinct autotuner key + outer runner so FC12 tactics do
    # not collide with the parent CuteDSL backend, and (2) drives the fused
    # single-op path in ``run_moe_nvfp4_impl``. uGPU is never used here.
    # ------------------------------------------------------------------
    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        weight_view: Optional[NvFp4WeightView] = None,
    ) -> torch.Tensor:
        assert self.has_nvfp4
        assert weight_view is not None
        output_dtype = torch.bfloat16

        if moe_output is None:
            moe_output = torch.empty(
                (token_final_scales.size(0), self.hidden_size), dtype=output_dtype, device=x.device
            )
        else:
            assert moe_output.size() == (token_final_scales.size(0), self.hidden_size)
            assert moe_output.dtype == output_dtype

        # Empty micro-batches: skip autotuning (synthetic grouped-GEMM inputs
        # require at least one output row).
        if token_selected_experts.size(0) == 0:
            return moe_output

        effective_top_k = token_selected_experts.size(-1)
        tuner = AutoTuner.get()
        runner = CuteDslFc12FusedMoENvfp4Runner(
            forward_impl=self.run_moe_nvfp4_impl,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=weight_view.expert_size_per_partition,
            local_expert_offset=weight_view.slot_start,
            enable_finalize_fusion=self.use_fused_finalize,
            enable_alltoall=enable_alltoall,
        )
        inputs = [x, token_selected_experts, token_final_scales, x_sf, moe_output, weight_view]
        _, best_tactic = tuner.choose_one(
            "CuteDslFc12FusedMoE::run_moe_nvfp4",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def run_moe_nvfp4_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weight_view: NvFp4WeightView,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        """Single fused FC1+FC2 op (replaces the parent's two-op sequence)."""
        effective_top_k = token_selected_experts.size(1)
        esp = weight_view.expert_size_per_partition
        slot_start = weight_view.slot_start

        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
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

        # One fused op: gather + FC1 GEMM + SwiGLU + requant + FC2 GEMM +
        # finalize (scatter-add into moe_output = a2a combine workspace).
        # fc1_alpha/fc2_alpha map 1:1 to the two-op path's per-expert global
        # scales (the kernel takes split alphas). The three atomic counters are
        # allocated + memset inside the op runner. The moe_output zeroing memset
        # is issued inside the op too (right before the fused kernel, so it is
        # the kernel's immediate stream predecessor and the PDL prologue can
        # overlap it); expanded_idx_to_permuted_idx / ep_size / enable_alltoall
        # are passed through for that memset.
        torch.ops.trtllm.cute_dsl_nvfp4_fc12_fused_rubin(
            input=x.view(torch.float4_e2m1fn_x2),
            fc1_weight=weight_view.w3_w1_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            fc1_weight_scale=weight_view.fc1_weight_scale.view(torch.uint8),
            fc1_alpha=weight_view.fc1_global_scale,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=self.fc2_input_scale,
            fc2_weight=weight_view.w2_weight.view(torch.float4_e2m1fn_x2),
            fc2_weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
            fc2_alpha=weight_view.fc2_global_scale,
            output=moe_output,
            token_final_scales=token_final_scales,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
            swiglu_limit=self.act_clamp,
            ep_size=self.mapping.moe_ep_size,
            enable_alltoall=enable_alltoall,
            scaling_vector_size=16,
        )
        return moe_output
