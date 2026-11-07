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

FC1+FC2-fused CuteDSL MXFP8 MoE backend for Rubin (SM107) and Blackwell
(SM100/SM103). One persistent kernel
(``trtllm::cute_dsl_mxfp8_fused_fc12_moe_inplace_rubin``, or its SM100 port
``..._blackwell``) does gather + FC1 GEMM + SwiGLU + MXFP8 requant + FC2 GEMM +
top-k finalize (scatter-add into ``moe_output``): W8A8 e4m3 x e4m3 with UE8M0
1x32 block scales. With locality domains (Rubin), the planner instead splits
every expert's inner channels across the two domains for every shape of the
layer; the shards are then the only copy of the expert weights.

Its weights are the Cutlass MXFP8 layout plus a gate/up interleave of FC1
(``MXFP8CuteDslFusedMoEMethod``). The kernel is a separate copy of the fused
FC12 kernel the NVFP4 leaf runs, so the two share no code here.
"""

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .....autotuner import AutoTuner
from .....cute_dsl_utils import (
    MXFP8_FUSED_FC12_RESET_MAX_ELEMENTS,
    mxfp8_fused_fc12_tuner_key_suffix,
)
from .....locality_domain.policy import PartitionPlan
from .....locality_domain_utils import (
    _copy_to_new_cuda_allocation,
    get_locality_domain_compute_sm_counts,
)
from .....model_config import ModelConfig
from .....utils import ActivationType, AuxStreamType, EventType, swizzle_sf, unswizzle_sf
from ...activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ...cutlass.eligibility import BF16_ONLY, SmSupport, check_dtype, check_sm
from ...impl_base import MoEImplBase, apply_moe_impl_construction_state, moe_tune_max_num_tokens
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
        doc="Rubin (SM107) / Blackwell (SM100/SM103) MXFP8 fused FC1 + SwiGLU + FC2 + "
        "finalize persistent CuTe DSL grouped GEMM.",
    )

    sm_support = SmSupport(allowed=frozenset({100, 103, 107}))
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

    # Released after the locality-domain split together with the full
    # weights: the shards carry their own swizzled block scales.
    _locality_domain_full_scale_params = ("w3_w1_weight_scale", "w2_weight_scale")

    # Raw BF16/FP16 activations with at most this many elements (4096 rows of
    # hidden 4096) are quantized inside the FC12 workspace-reset launch
    # instead of a separate ``mxfp8_quantize`` launch; larger, prefill-sized
    # inputs keep the native launch.
    FUSED_INPUT_QUANT_MAX_ELEMENTS = 1 << 24

    # The workspace reset indexes the output with Int32, so larger outputs
    # are cleared by the memset instead.
    _ZERO_OUTPUT_MAX_ELEMENTS = MXFP8_FUSED_FC12_RESET_MAX_ELEMENTS

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """MXFP8 on Rubin (SM107) or Blackwell (SM100/SM103), with a CuTe DSL build
        that has the fused FC12 kernel for that architecture."""
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

        # Older Rubin CuTe DSL builds miscompile the kernel's MXFP8 path; the
        # SM100 port needs the same submodules plus the Blackwell helpers.
        if d.env.sm == 107:
            deps = (MoEDep.CUTEDSL_RUBIN, MoEDep.CUTEDSL_MXFP8_FUSED_FC12)
            build = "internal 0.3.0+20260803"
        else:
            deps = (MoEDep.CUTEDSL_MXFP8_FUSED_FC12_BLACKWELL,)
            build = "4.8.0a0+20260821"
        for dep in deps:
            if not d.env.has_dep(dep):
                return _reject(
                    MoERejectReason.DEP_MISSING,
                    f"{cls.__name__} on SM{d.env.sm} requires a CuTe DSL build with the "
                    f"fused FC12 kernel's submodules (e.g. {build} or newer)",
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
        # Anchors the autotuning buckets (see
        # CuteDslFusedMoEMxfp8Runner.get_tuning_config): every power-of-two
        # bucket up to this count is profiled, so a small warmup forward still
        # tunes the larger buckets, and profiling stops at this cap.
        self.tune_max_num_tokens = moe_tune_max_num_tokens(
            model_config, self.num_slots, routing_method.get_experts_per_token()
        )
        # SM107 runs the Rubin kernel, SM100/SM103 its Blackwell port; both ops
        # share the input contract.
        self._sm_version = get_sm_version()
        self._init_output_memset_and_locality_domain(model_config)
        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self) -> None:
        if self._weights_created:
            return
        self._check_quant_config_is_my_format()
        super().create_weights()

    def _adjust_locality_domain_plan(self, plan: PartitionPlan) -> PartitionPlan:
        """Veto a split the shard kernels cannot run.

        Each shard takes whole 128-channel groups, and the fused kernel needs
        every persistent 2-CTA cluster of its grid resident at once (FC2 waits
        on FC1 readiness), so each partition must hold a whole number of
        clusters.
        """
        if not plan.enabled:
            return plan
        reason = None
        granule = self._INTERMEDIATE_ALIGNMENT * plan.num_partitions
        if self.intermediate_size_per_partition % granule != 0:
            reason = (
                "MXFP8 locality domain MoE needs intermediate_size_per_partition "
                f"({self.intermediate_size_per_partition}) to split into "
                f"{plan.num_partitions} multiples of {self._INTERMEDIATE_ALIGNMENT} "
                "channels"
            )
        else:
            for pid in range(plan.num_partitions):
                sm_counts = get_locality_domain_compute_sm_counts(pid)
                if sm_counts is None or sm_counts[0] % 2 != 0:
                    reason = (
                        f"MXFP8 locality domain MoE needs partition {pid} to hold whole "
                        f"2-CTA clusters, got SM counts {sm_counts}"
                    )
                    break
        if reason is None:
            return plan
        reason += "; running the full-GPU fused FC12 kernel"
        logger.warning_once(f"{type(self).__name__}: {reason}", key=reason)
        return PartitionPlan(enabled=False, reason_if_disabled=reason)

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
    def _fuses_input_quant(self, x: torch.Tensor) -> bool:
        """Whether ``x`` is quantized inside the fused FC12 workspace reset.

        The reset kernel reads the input with aligned vector loads and writes
        one E8M0 scale per 32 elements in linear order. ``can_implement``
        already requires ``hidden % 512 == 0``, so the hidden size needs no
        check here.

        Only the Rubin kernel fuses the quantization into its reset launch;
        the Blackwell op quantizes raw input in a separate launch, so on
        SM100/SM103 the native quantizer is used directly.
        """
        return (
            self._sm_version == 107
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.dim() == 2
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and x.numel() <= self.FUSED_INPUT_QUANT_MAX_ELEMENTS
        )

    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic MXFP8 activation quantization (e4m3 + UE8M0 1x32 block scales).

        The fused FC12 kernel gathers the per-token scales row by row, so they
        are always produced in the linear layout, which is also what post-quant
        communication transports. With ``post_quant_comm=False`` (no
        communication, or after a pre-quant dispatch), a small enough input is
        left raw: the full-GPU op and the locality-domain split both quantize
        it in their (single) workspace-reset launch, saving the separate
        quantize launch.

        Args:
            x: ``[num_tokens, hidden_size]`` bfloat16 activations.
            post_quant_comm: Whether the result is dispatched; E4M3 travels on
                the wire, so it is always quantized here.

        Returns:
            ``(x, x_sf)``: e4m3 ``[num_tokens, hidden_size]`` and uint8
            ``[num_tokens, hidden_size // 32]`` scales; or, when quantization is
            deferred to the op, ``x`` unchanged and an uninitialized scale
            tensor of that shape, which carries shape metadata only.
        """
        x_row = x.shape[0]
        sf_vec_size = self.quant_method.BLOCK_SIZE
        if not post_quant_comm and self._fuses_input_quant(x):
            x_sf = torch.empty(
                (x_row, x.shape[1] // sf_vec_size), dtype=torch.uint8, device=x.device
            )
            return x, x_sf
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

        effective_top_k = token_selected_experts.size(-1)
        tuner = AutoTuner.get()
        # The locality-domain split is a static planner decision
        # (``enable_locality_domains`` / ``plan_moe``), not a per-shape runtime
        # choice: the full weights are released after sharding, so only the
        # shards exist and every shape runs the split.
        use_locality_domain = self._locality_domain_runtime is not None
        if use_locality_domain:
            assert self._locality_domain_weight_shards is not None, (
                "locality-domain MXFP8 MoE needs transform_weights before the first forward"
            )
            runner = CuteDslFusedMoEMxfp8Runner(
                forward_impl=self._run_moe_mxfp8_locality_domain,
                num_experts=self.num_slots,
                top_k=effective_top_k,
                num_local_experts=self.expert_size_per_partition,
                local_expert_offset=self.slot_start,
                enable_alltoall=enable_alltoall,
                tune_max_num_tokens=self.tune_max_num_tokens,
                workload_identity=self._locality_domain_workload_identity(x.dtype),
            )
        else:
            runner = CuteDslFusedMoEMxfp8Runner(
                forward_impl=self.run_moe_mxfp8_impl,
                num_experts=self.num_slots,
                top_k=effective_top_k,
                num_local_experts=self.expert_size_per_partition,
                local_expert_offset=self.slot_start,
                enable_alltoall=enable_alltoall,
                tune_max_num_tokens=self.tune_max_num_tokens,
            )
        inputs = [x, token_selected_experts, token_final_scales, x_sf, moe_output]
        _, best_tactic = tuner.choose_one(
            self._tuner_key(x.dtype, locality=use_locality_domain),
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    @staticmethod
    def _tuner_key(dtype: torch.dtype, locality: bool = False) -> str:
        """Separate quantize/reset timing from the prequantized op cache."""
        key = "CuteDslFusedMoE::run_moe_mxfp8"
        if locality:
            key += "::locality_domain"
        return key + mxfp8_fused_fc12_tuner_key_suffix(dtype)

    def _memset_moe_output(
        self,
        moe_output: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_size: int,
        top_k: int,
        enable_alltoall: bool,
    ) -> None:
        """Zero the rows this rank finalizes.

        It runs on the aux stream when one is set up, but it reads
        ``moe_sort``'s outputs and the main stream waits for it right away, so
        nothing overlaps it.
        """
        memset_kwargs = dict(
            input=moe_output,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_tokens_dim=tile_size,
            top_k=top_k,
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

    def _run_moe_mxfp8_locality_domain(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        """MXFP8 MoE with the fused FC12 kernel split across two locality domains.

        Each shard holds half of every expert's inner channels (FC1 columns as
        whole gate/up pairs, FC2 rows of K) and both scatter-add their partial
        FC2 result into the shared ``moe_output``, which the op's parent reset
        launch clears first (dense case). Routing metadata is shared and raw
        activations are quantized by that same launch. The split changes the
        BF16 accumulation order relative to the full-GPU kernel.
        """
        effective_top_k = token_selected_experts.size(1)
        esp = self.expert_size_per_partition
        slot_start = self.slot_start
        shards = self._locality_domain_weight_shards

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

        # Same rule as run_moe_mxfp8_impl: in the dense case the op's parent
        # reset launch clears the whole output for both shards before the
        # fork; alltoall with ep > top_k, or an output past the reset kernel's
        # Int32 element count, keeps the row memset.
        zero_output_in_kernel = (
            not enable_alltoall or self.mapping.moe_ep_size <= effective_top_k
        ) and moe_output.numel() <= self._ZERO_OUTPUT_MAX_ELEMENTS
        if not zero_output_in_kernel:
            self._memset_moe_output(
                moe_output,
                tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                tile_size,
                effective_top_k,
                enable_alltoall,
            )

        fc_alpha, fc1_norm_const = self._fused_fc12_constants(x.device)
        torch.ops.trtllm.cute_dsl_mxfp8_fused_fc12_moe_locality_domain_inplace_rubin(
            input=x,
            fc1_weight_0=shards[0]["w3_w1_weight"],
            fc1_weight_1=shards[1]["w3_w1_weight"],
            input_scale=x_sf,
            fc1_weight_scale_0=shards[0]["fc31_weight_block_scale"].view(torch.uint8),
            fc1_weight_scale_1=shards[1]["fc31_weight_block_scale"].view(torch.uint8),
            fc1_alpha=fc_alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            fc1_norm_const=fc1_norm_const,
            fc2_weight_0=shards[0]["w2_weight"],
            fc2_weight_1=shards[1]["w2_weight"],
            fc2_weight_scale_0=shards[0]["fc2_weight_block_scale"].view(torch.uint8),
            fc2_weight_scale_1=shards[1]["fc2_weight_block_scale"].view(torch.uint8),
            fc2_alpha=fc_alpha,
            output=moe_output,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
            swiglu_limit=self.act_clamp,
            zero_output=zero_output_in_kernel,
        )
        return moe_output

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
        # rank produces must be zero on entry. In the dense case (the same
        # condition under which moe_output_memset_inplace falls back to a full
        # cudaMemsetAsync) the fused op clears the whole output inside its
        # workspace reset launch, saving the memset launch and its aux-stream
        # event pair. Alltoall with ep > top_k keeps the sparse row memset,
        # and so does an output past the reset kernel's Int32 element count.
        zero_output_in_kernel = (
            not enable_alltoall or self.mapping.moe_ep_size <= effective_top_k
        ) and moe_output.numel() <= self._ZERO_OUTPUT_MAX_ELEMENTS
        if not zero_output_in_kernel:
            self._memset_moe_output(
                moe_output,
                tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                tile_size,
                effective_top_k,
                enable_alltoall,
            )

        fc_alpha, fc1_norm_const = self._fused_fc12_constants(x.device)
        fused_fc12_op = (
            torch.ops.trtllm.cute_dsl_mxfp8_fused_fc12_moe_inplace_rubin
            if self._sm_version == 107
            else torch.ops.trtllm.cute_dsl_mxfp8_fused_fc12_moe_inplace_blackwell
        )
        # The block scales are stored int32-packed (4 UE8M0 per int32 along
        # K); the kernel reads them as a [E, N, K/32] byte matrix.
        fused_fc12_op(
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
            zero_output=zero_output_in_kernel,
        )
        return moe_output

    # ------------------------------------------------------------------
    # Locality domain
    # ------------------------------------------------------------------
    def pre_reload_weights(self) -> None:
        """Recreate the full weights from the shards before a reload.

        The split released the full weights, and the base reload re-creates
        them uninitialized. A refit may rewrite only part of the layer (one
        projection, some experts, or nothing), so copy the shards back into
        them first: the untouched parts then keep their current values. The
        shard tensors are kept, and the next split refreshes them in place,
        so captured CUDA graphs keep reading valid addresses.
        """
        shards = self._locality_domain_weight_shards
        super().pre_reload_weights()
        if shards is not None:
            self._restore_full_weights_from_shards(shards)
            self._locality_domain_shard_buffers = shards

    @torch.no_grad()
    def _restore_full_weights_from_shards(self, shards: list[dict[str, torch.Tensor]]) -> None:
        """Write the kernel-layout full weights and block scales back from the shards."""
        sf_vec = self.quant_method.BLOCK_SIZE
        num_experts, hidden, inner = self.w2_weight.shape
        half_inner = inner // len(shards)
        self.w3_w1_weight.data.copy_(torch.cat([s["w3_w1_weight"] for s in shards], dim=1))
        self.w2_weight.data.copy_(torch.cat([s["w2_weight"] for s in shards], dim=2))
        # The FC1 shards cut whole 128-row swizzle atoms, so their scales are
        # row slices of the swizzled tensor.
        fc1_scale = torch.cat([s["fc31_weight_block_scale"] for s in shards], dim=1)
        self.w3_w1_weight_scale.data.copy_(fc1_scale.view(self.w3_w1_weight_scale.shape))
        # The FC2 shards cut the K/32 columns, which the swizzle interleaves:
        # rebuild the linear matrix and swizzle it again.
        fc2_scale_linear = torch.cat(
            [
                unswizzle_sf(
                    s["fc2_weight_block_scale"].view(torch.uint8), hidden, half_inner, sf_vec
                ).view(num_experts, hidden, half_inner // sf_vec)
                for s in shards
            ],
            dim=2,
        )
        fc2_scale = swizzle_sf(fc2_scale_linear.contiguous(), hidden, inner, sf_vec)
        self.w2_weight_scale.data.copy_(
            fc2_scale.view(torch.uint8)
            .view(self.w2_weight_scale.dtype)
            .view(self.w2_weight_scale.shape)
        )
        self.quant_method.mark_all_slots_in_kernel_layout(self)
        self.quant_method.setup_quant_scales(self)

    def _split_weights_for_locality_domain(self) -> list[dict[str, torch.Tensor]]:
        """Split every expert's inner channels for the fused FC12 kernel.

        FC1 weights ``[E, 2I, K]`` are sliced along N: the gate/up interleave
        has granularity 64, so each half keeps whole pairs. FC2 weights
        ``[E, K, I]`` are sliced along the inner (K) axis. The block scales are
        stored in the 128x4 swizzled layout (int32-packed), whose memory order
        is not the logical ``[N, K/32]`` matrix, so they are unswizzled, sliced
        like their weights and swizzled again per shard. The two shards
        together hold one copy of the weights. After a reload, the previous
        shard tensors are refreshed in place (see :meth:`pre_reload_weights`).
        """
        num_p = self._locality_domain_plan.num_partitions
        if num_p != 2:
            raise ValueError(f"MXFP8 locality domain MoE supports two partitions, got {num_p}")
        sf_vec = self.quant_method.BLOCK_SIZE
        hidden = self.w2_weight.size(1)
        inner = self.w2_weight.size(2)
        fc1_n = self.w3_w1_weight.size(1)
        assert fc1_n == 2 * inner, (fc1_n, inner)
        half_inner = inner // num_p
        num_experts = self.w2_weight.size(0)
        fc1_scale = self.quant_scales.fc31_weight_block_scale
        fc2_scale = self.quant_scales.fc2_weight_block_scale
        assert fc1_scale.size(1) == fc1_n and fc2_scale.size(1) == hidden, (
            fc1_scale.shape,
            fc2_scale.shape,
        )
        # Linear [E, rows, cols / 32] byte views of the swizzled scales.
        fc1_scale_linear = unswizzle_sf(fc1_scale.view(torch.uint8), fc1_n, hidden, sf_vec).view(
            num_experts, fc1_n, hidden // sf_vec
        )
        fc2_scale_linear = unswizzle_sf(fc2_scale.view(torch.uint8), hidden, inner, sf_vec).view(
            num_experts, hidden, inner // sf_vec
        )
        previous = getattr(self, "_locality_domain_shard_buffers", None)
        self._locality_domain_shard_buffers = None
        shards = []
        for pid in range(num_p):
            n_slice = slice(pid * 2 * half_inner, (pid + 1) * 2 * half_inner)
            k_slice = slice(pid * half_inner, (pid + 1) * half_inner)
            sk_slice = slice(pid * half_inner // sf_vec, (pid + 1) * half_inner // sf_vec)
            fc1_shard_scale = (
                swizzle_sf(
                    fc1_scale_linear[:, n_slice].contiguous(), 2 * half_inner, hidden, sf_vec
                )
                .view(num_experts, 2 * half_inner, hidden // sf_vec)
                .view(fc1_scale.dtype)
            )
            fc2_shard_scale = (
                swizzle_sf(
                    fc2_scale_linear[:, :, sk_slice].contiguous(), hidden, half_inner, sf_vec
                )
                .view(num_experts, hidden, half_inner // sf_vec)
                .view(fc2_scale.dtype)
            )
            sources = {
                "w3_w1_weight": self.w3_w1_weight[:, n_slice],
                "w2_weight": self.w2_weight[:, :, k_slice],
                "fc31_weight_block_scale": fc1_shard_scale,
                "fc2_weight_block_scale": fc2_shard_scale,
            }
            with self._locality_domain_runtime.partition_weight_context(pid):
                old = previous[pid] if previous is not None else None
                if old is not None and all(
                    old[name].shape == src.shape and old[name].dtype == src.dtype
                    for name, src in sources.items()
                ):
                    for name, src in sources.items():
                        old[name].copy_(src)
                    shards.append(old)
                else:
                    shards.append(
                        {name: _copy_to_new_cuda_allocation(src) for name, src in sources.items()}
                    )
        return shards
