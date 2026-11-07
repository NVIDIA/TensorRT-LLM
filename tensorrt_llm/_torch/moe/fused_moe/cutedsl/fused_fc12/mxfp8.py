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
"""TrtllmCutedslFusedFc12Mxfp8Impl: ``trtllm.cutedsl.fused_fc12.mxfp8``.

FC1+FC2-fused CuteDSL MXFP8 MoE backend for Rubin (SM107). One persistent kernel
(``trtllm::cute_dsl_mxfp8_fused_fc12_moe_inplace_rubin``) does gather + FC1
GEMM + SwiGLU + MXFP8 requant + FC2 GEMM + top-k finalize (scatter-add into
``moe_output``): W8A8 e4m3 x e4m3 with UE8M0 1x32 block scales.

Its weights are the Cutlass MXFP8 layout plus a gate/up interleave of FC1
(``MXFP8CuteDslFusedMoEMethod``). The kernel is a separate copy of the fused
FC12 kernel the NVFP4 leaf runs, so the two share no code here.
"""

import torch

from .....autotuner import AutoTuner
from .....model_config import ModelConfig
from .....utils import ActivationType, AuxStreamType, EventType
from ...activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ...cutlass.eligibility import BF16_ONLY, SmSupport, check_dtype, check_sm
from ...impl_base import MoEImplBase, apply_moe_impl_construction_state
from ...impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    check_quant_matches_identity,
    require_comm_plan,
)
from ...impl_environment import MoEDep
from ...impl_identity import MoEImplDescriptor, MoEImplId, register_moe_impl
from ...interface import MoESchedulerKind, _reject
from ...quantization import MoEWeightLoadingMode, MXFP8CuteDslFusedMoEMethod
from ...routing import BaseMoeRoutingMethod
from ..identity import KERNEL_FUSED_FC12, PROVIDER_TRTLLM, TECHNIQUE_CUTEDSL
from ..kernel_inputs import finalize_routing_scales
from ..locality_domain import CuteDslLocalityDomainMixin
from ..runners import CuteDslFusedMoEMxfp8Runner


@register_moe_impl
class TrtllmCutedslFusedFc12Mxfp8Impl(CuteDslLocalityDomainMixin, MoEImplBase):
    """``trtllm.cutedsl.fused_fc12.mxfp8``: FC1+FC2-fused CuteDSL MXFP8 MoE.

    Takes the constructor arguments of ``CuteDslFusedMoEBase``.
    """

    # The fused kernel takes per-tile expert ranges, so EP need not divide the
    # expert count.
    _supports_non_divisible_ep: bool = True

    descriptor = MoEImplDescriptor(
        identity=MoEImplId(PROVIDER_TRTLLM, TECHNIQUE_CUTEDSL, KERNEL_FUSED_FC12, "mxfp8"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=MoEStaticCapability(
            supports_apply_router_weight_on_input=True,
        ),
        # The persistent FC2 tiles wait on FC1 tiles of the same grid.
        input_requirement=MoEInputRequirement(
            routing_scales_dtype=torch.float32,
            run_moe_requires_coresident_grid=True,
        ),
        doc="Rubin (SM107) MXFP8 fused FC1 + SwiGLU + FC2 + finalize persistent "
        "CuTe DSL grouped GEMM.",
    )

    sm_support = SmSupport(allowed=frozenset({107}))
    # The kernel writes a bfloat16 output.
    supported_dtypes = BF16_ONLY

    # The epilogue is SwiGLU with an optional clamp, which the kernel always
    # applies, so a clamp-less layer passes +inf.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu}),
        limit=ActivationParamShape.UNIFORM_SCALAR,
        limit_when_absent=float("inf"),
    )

    # The FC1 scale-factor gather reads 16-byte chunks of UE8M0 scales per
    # row, i.e. 16 * 32 = 512 K elements.
    _HIDDEN_ALIGNMENT = MXFP8CuteDslFusedMoEMethod.FC1_K_ALIGNMENT
    # FC1 N / FC2 K per rank: 128-row scale atoms and int32-packed scales.
    _INTERMEDIATE_ALIGNMENT = 128

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """MXFP8 on Rubin (SM107) with a CuTe DSL build that has the fused FC12 kernel."""
        for verdict in (
            check_quant_matches_identity(cls, p),
            check_sm(cls, d),
            check_dtype(cls, p),
        ):
            if verdict is not None:
                return verdict

        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support swiglu_gptoss_style",
            )

        if p.bias:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} has no expert bias input",
            )

        if p.hidden_size is not None and p.hidden_size % cls._HIDDEN_ALIGNMENT != 0:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"{cls.__name__} needs hidden_size % {cls._HIDDEN_ALIGNMENT} == 0, "
                f"got {p.hidden_size}",
            )

        # The MXFP8 weights pack four UE8M0 scales per int32 along K, and the
        # kernel tiles FC1 N / FC2 K in 128-row atoms, so each rank's slice of
        # the intermediate size must be a multiple of 128. Uneven shards are a
        # separate concern and are not answered here.
        tp_size = max(d.tp_size, 1)
        if p.intermediate_size is not None and p.intermediate_size % tp_size == 0:
            per_partition = p.intermediate_size // tp_size
            if per_partition % cls._INTERMEDIATE_ALIGNMENT != 0:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED,
                    f"{cls.__name__} needs intermediate_size_per_partition % "
                    f"{cls._INTERMEDIATE_ALIGNMENT} == 0, got {per_partition} "
                    f"(intermediate_size={p.intermediate_size}, moe_tp_size={tp_size})",
                )

        # MXFP8 weights cannot follow EPLB expert migration
        # (MXFP8CutlassFusedMoEMethod.eplb_support_status).
        if d.eplb_enabled:
            return _reject(
                MoERejectReason.EPLB_UNSUPPORTED,
                f"{cls.__name__} does not support EPLB",
            )

        # The fused kernel scatter-adds FC2 straight into moe_output; there is
        # no unfused FC2 variant.
        if not d.fused_finalize_enabled:
            return _reject(
                MoERejectReason.FINALIZE_FUSION_REQUIRED,
                f"{cls.__name__} only has a fused-finalize FC2",
            )

        # Older Rubin CuTe DSL builds miscompile the kernel's MXFP8 path.
        for dep in (MoEDep.CUTEDSL_RUBIN, MoEDep.CUTEDSL_MXFP8_FUSED_FC12):
            if not d.env.has_dep(dep):
                return _reject(
                    MoERejectReason.DEP_MISSING,
                    f"{cls.__name__} requires a CuTe DSL build with the "
                    "cutlass.memory / cutlass.tensor_utils submodules (e.g. internal "
                    "0.3.0+20260803 or newer)",
                )

        return MoEEligibility.ok()

    # ------------------------------------------------------------------
    # Construction and weights.
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: int | None = None,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ) -> None:
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
        # the locality-domain planner reads the flag.
        self.use_fused_finalize = (
            not model_config.moe_disable_finalize_fusion and model_config.lora_config is None
        )
        self._init_output_memset_and_locality_domain(model_config)
        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self) -> None:
        if self._weights_created:
            return
        self._check_quant_config_is_my_format()
        super().create_weights()

    def _get_quant_method(self) -> object:
        # Cutlass MXFP8 storage/swizzle + the gate/up interleave the fused FC12
        # kernel expects.
        return MXFP8CuteDslFusedMoEMethod()

    def _check_configs(self) -> None:
        assert self._weights_created
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

    def _supports_load_balancer(self) -> bool:
        # The fused op takes precomputed routing (token_selected_experts and
        # token_final_scales), so the scheduler must route before run_moe even
        # without communication or attention DP. EPLB itself is declined in
        # can_implement.
        return True

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        # The fused op finalizes straight into moe_output, which may be the
        # all-to-all combine workspace.
        return True

    # ------------------------------------------------------------------
    # Input quantization and dispatch
    # ------------------------------------------------------------------
    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic MXFP8 activation quantization (e4m3 + UE8M0 1x32 block scales).

        The fused FC12 kernel gathers the per-token scales row by row, so they
        are always produced in the linear layout, which is also what post-quant
        communication transports.

        Args:
            x: ``[num_tokens, hidden_size]`` bfloat16 activations.
            post_quant_comm: Ignored; both paths produce the same layout.

        Returns:
            ``(x, x_sf)``: e4m3 ``[num_tokens, hidden_size]`` and uint8
            ``[num_tokens, hidden_size // 32]`` scales.
        """
        del post_quant_comm
        x_row = x.shape[0]
        sf_vec_size = self.quant_method.BLOCK_SIZE
        x, x_sf = torch.ops.trtllm.mxfp8_quantize(
            x, False, alignment=self.quant_method.weight_alignment
        )
        # ``view(0, -1)`` is ambiguous for an empty micro-batch, so spell out
        # the scale width.
        return x, x_sf.view(x_row, self.hidden_size // sf_vec_size)

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # the fused kernel allocates its own intermediates
        plan = require_comm_plan(self, ctx)
        return self.run_moe_mxfp8(
            x=ctx.x,
            token_selected_experts=ctx.token_selected_experts,
            token_final_scales=finalize_routing_scales(self, ctx),
            x_sf=ctx.x_sf,
            moe_output=plan.moe_output,
            enable_alltoall=plan.enable_alltoall,
        )

    def _fused_fc12_constants(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (per-expert alpha, FC1 norm const) inputs of the kernel.

        MXFP8 carries all scaling in the UE8M0 block scales, so both the
        per-expert dequantization alphas and the FC1 output normalization
        constant are exactly 1. They are kernel inputs nonetheless (the kernel
        also serves NVFP4), so keep one cached copy per module rather than
        allocating on every forward.
        """
        num_local_experts = self.expert_size_per_partition
        cached = getattr(self, "_fused_fc12_consts", None)
        if cached is None or cached[0].device != device or cached[0].numel() != num_local_experts:
            cached = (
                torch.ones(num_local_experts, dtype=torch.float32, device=device),
                torch.ones(1, dtype=torch.float32, device=device),
            )
            self._fused_fc12_consts = cached
        return cached

    def run_moe_mxfp8(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor | None = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        """Autotuner wrapper: picks the routing tile and runs the fused op."""
        assert x_sf is not None, "MXFP8 MoE requires the activation block scales"
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

        tuner = AutoTuner.get()
        runner = CuteDslFusedMoEMxfp8Runner(
            forward_impl=self.run_moe_mxfp8_impl,
            num_experts=self.num_slots,
            top_k=token_selected_experts.size(-1),
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            enable_alltoall=enable_alltoall,
        )
        inputs = [x, token_selected_experts, token_final_scales, x_sf, moe_output]
        _, best_tactic = tuner.choose_one(
            "CuteDslFusedMoE::run_moe_mxfp8",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def run_moe_mxfp8_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        """MXFP8 MoE via the fused FC1+FC2 CuTe DSL kernel."""
        effective_top_k = token_selected_experts.size(1)
        esp = self.expert_size_per_partition
        slot_start = self.slot_start

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

        # The fused finalize scatter-adds into moe_output, so the rows this
        # rank produces must be zero on entry. Overlap the sparse memset with
        # the routing metadata on the aux stream when one is available.
        memset_kwargs = dict(
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
        if self._has_moe_output_memset_aux_stream():
            memset_stream = self._moe_output_memset_run_stream()
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(memset_stream)
            with torch.cuda.stream(memset_stream):
                self.event_dict[EventType.Main].wait()
                torch.ops.trtllm.moe_output_memset_inplace(**memset_kwargs)
                self.event_dict[EventType.MoeOutputMemset].record()
            self.event_dict[EventType.MoeOutputMemset].wait()
        else:
            torch.ops.trtllm.moe_output_memset_inplace(**memset_kwargs)

        fc_alpha, fc1_norm_const = self._fused_fc12_constants(x.device)
        # The block scales are stored int32-packed (4 UE8M0 per int32 along
        # K); the kernel reads them as a [E, N, K/32] byte matrix.
        torch.ops.trtllm.cute_dsl_mxfp8_fused_fc12_moe_inplace_rubin(
            input=x,
            fc1_weight=self.w3_w1_weight,
            input_scale=x_sf,
            fc1_weight_scale=self.quant_scales.fc31_weight_block_scale.view(torch.uint8),
            fc1_alpha=fc_alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            fc1_norm_const=fc1_norm_const,
            fc2_weight=self.w2_weight,
            fc2_weight_scale=self.quant_scales.fc2_weight_block_scale.view(torch.uint8),
            fc2_alpha=fc_alpha,
            output=moe_output,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
            swiglu_limit=self.act_clamp,
        )
        return moe_output
