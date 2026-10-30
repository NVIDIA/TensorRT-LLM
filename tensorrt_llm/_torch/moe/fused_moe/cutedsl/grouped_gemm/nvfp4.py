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
"""``TrtllmCutedslGroupedGemmNvfp4Impl``: ``trtllm.cutedsl.grouped_gemm.nvfp4``."""

import torch

from tensorrt_llm._utils import get_env_flag, get_sm_version

from .....autotuner import AutoTuner
from .....cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from .....locality_domain_utils import _copy_to_new_cuda_allocation
from .....utils import ActivationType, AuxStreamType, EventType, Fp4QuantizedTensor
from ...activation import ActivationParamShape, MoEActivationSupport
from ...cutlass.eligibility import SmSupport
from ...impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERunContext,
    MoEStaticCapability,
    nvfp4_fc1_row_alignment_rejection,
    require_comm_plan,
)
from ...impl_identity import MoEImplDescriptor, register_moe_impl
from ...interface import MoESchedulerKind
from ...quantization import NVFP4CuteDslFusedMoEMethod
from ..identity import grouped_gemm_identity
from ..kernel_inputs import NvFp4WeightView, finalize_routing_scales, quantize_input_nvfp4
from ..runners import CuteDslFusedMoENvfp4Runner
from .base import CuteDslFusedMoEBase
from .eligibility import check_grouped_gemm_leaf

# Kill switch for TrtllmCutedslGroupedGemmNvfp4Impl.quantize_input_async.
ASYNC_INPUT_QUANT = get_env_flag("TLLM_MOE_ASYNC_INPUT_QUANT", True)
# Largest batch quantize_input_async moves to the aux stream. Only
# latency-bound decode batches gain from the overlap; on a prefill batch the
# cross-stream record_stream of its large input and output keeps the caching
# allocator from reusing those blocks and forces synchronizing cudaFree calls.
ASYNC_INPUT_QUANT_MAX_TOKENS = 1024

_DISABLE_DIRECT_DEEP_EP_METADATA_ENV = "TRTLLM_DISABLE_CUTEDSL_DEEP_EP_DIRECT_METADATA"


@register_moe_impl
class TrtllmCutedslGroupedGemmNvfp4Impl(CuteDslFusedMoEBase):
    """``trtllm.cutedsl.grouped_gemm.nvfp4``: CuteDSL NVFP4 grouped GEMM.

    FC2's finalize is fused, or unfused off SM107 when finalize fusion is
    disabled. Admits the NVFP4_AWQ / NVFP4_ARC calibration aliases, which
    ``MoEProblem.identity_quant`` folds onto ``nvfp4``.
    """

    descriptor = MoEImplDescriptor(
        identity=grouped_gemm_identity("nvfp4"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=MoEStaticCapability(
            supports_dwdp=True,
            supports_eplb=True,
            supports_apply_router_weight_on_input=True,
            supports_deep_ep_direct_metadata=True,
        ),
        input_requirement=MoEInputRequirement(routing_scales_dtype=torch.float32),
        doc="CuTe DSL NVFP4 grouped GEMM: gather + FC1 + fused activation, "
        "then FC2 + finalize; SM100/SM103/SM107.",
    )
    sm_support = SmSupport(allowed=frozenset({100, 103, 107}))
    _locality_domain_full_scale_params = ("w3_w1_weight_scale", "w2_weight_scale")

    # Kinds mirror the act-fusion kernels' own SUPPORTED_ACTIVATION_TYPES.
    # There are two of them and ``run_moe_nvfp4`` picks between them by SM:
    #   moe/kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py
    #   moe/kernels/rubin/rubin_contiguous_gather_grouped_blockscaled_gemm_act_fusion.py
    # alpha/beta carry SiTU's soft-caps. The clamp is a kernel-cache-key scalar
    # and the epilogue has no "clamp absent" branch, so an absent clamp is
    # +inf, not None.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Relu2, ActivationType.SiTu}),
        alpha_beta=ActivationParamShape.UNIFORM_SCALAR,
        limit=ActivationParamShape.UNIFORM_SCALAR,
        limit_when_absent=float("inf"),
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        # process_weights_after_loading() unswizzles the FC1 block scales,
        # which asserts 128-row tiles; without this gate an unaligned shard
        # dies mid weight load with a bare swizzle error.
        return check_grouped_gemm_leaf(cls, p, d, nvfp4_fc1_row_alignment_rejection(p, d))

    def _init_format_state(self) -> None:
        # Opt-out for adapter-free DeepEP metadata; see
        # can_use_deep_ep_direct_metadata.
        self.disable_deep_ep_direct_metadata = get_env_flag(
            _DISABLE_DIRECT_DEEP_EP_METADATA_ENV, False
        )
        # Async input quantization (see quantize_input_async /
        # _join_pending_x_ready).
        self._x_quant_event = torch.cuda.Event()
        self._pending_x_ready_event: torch.cuda.Event | None = None

    def _get_quant_method(self) -> object:
        return NVFP4CuteDslFusedMoEMethod()

    def can_use_deep_ep_direct_metadata(self, supports_post_quant_dispatch: bool) -> bool:
        """Report complete runtime eligibility for adapter-free DeepEP metadata."""
        return (
            not self.disable_deep_ep_direct_metadata
            and supports_post_quant_dispatch
            and self.use_fused_finalize
            and get_sm_version() != 107
            and getattr(self, "_locality_domain_runtime", None) is None
        )

    quantize_input = quantize_input_nvfp4

    def quantize_input_async(
        self,
        x: torch.Tensor,
        input_ready_event: torch.cuda.Event | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.cuda.Event | None]:
        """Quantize ``x`` on the MoE aux stream so it overlaps top-k and ``moe_sort``.

        At small decode batches the three per-layer prologue kernels (top-k,
        input quantization, routing/permutation) are all latency-bound and
        serialize on the main stream. Quantization only depends on ``x``, so
        the scheduler forks it here before routing; the consumer joins via the
        returned event right before the first kernel that reads ``x``/``x_sf``
        (``run_moe_nvfp4_impl``, after ``moe_sort``). Falls back to the
        synchronous ``quantize_input`` (event ``None``) when disabled, already
        quantized, larger than ``ASYNC_INPUT_QUANT_MAX_TOKENS`` tokens, or no
        aux stream is available. The stream running the quantization joins
        ``input_ready_event`` (``x`` written on another stream).
        """
        if (
            not ASYNC_INPUT_QUANT
            or isinstance(x, Fp4QuantizedTensor)
            or x.shape[0] > ASYNC_INPUT_QUANT_MAX_TOKENS
            or not self._has_moe_output_memset_aux_stream()
        ):
            if input_ready_event is not None:
                input_ready_event.wait()
            x_q, x_sf = self.quantize_input(x, post_quant_comm=False)
            return x_q, x_sf, None
        main_stream = torch.cuda.current_stream()
        aux_stream = self.aux_stream_dict[AuxStreamType.MoeOutputMemset]
        self.event_dict[EventType.Main].record()
        x.record_stream(aux_stream)
        with torch.cuda.stream(aux_stream):
            self.event_dict[EventType.Main].wait()
            if input_ready_event is not None:
                input_ready_event.wait()
            x_q, x_sf = self.quantize_input(x, post_quant_comm=False)
            self._x_quant_event.record()
        x_q.record_stream(main_stream)
        x_sf.record_stream(main_stream)
        return x_q, x_sf, self._x_quant_event

    def _join_pending_x_ready(self) -> None:
        """Wait for the async input quantization (once per forward)."""
        event = getattr(self, "_pending_x_ready_event", None)
        if event is not None:
            event.wait()
            self._pending_x_ready_event = None

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
        """Run the NVFP4 MoE computation; returns final_hidden_states."""
        del workspace  # CuteDSL kernels allocate their own intermediates.
        plan = require_comm_plan(self, ctx)
        return self.run_moe_nvfp4(
            x=ctx.x,
            token_selected_experts=ctx.token_selected_experts,
            token_final_scales=finalize_routing_scales(self, ctx),
            x_sf=ctx.x_sf,
            moe_output=plan.moe_output,
            enable_alltoall=plan.enable_alltoall,
            weight_view=self._build_local_weight_view(),
            recv_expert_count=plan.recv_expert_count,
            deep_ep_expert_capacity=plan.deep_ep_expert_capacity,
            use_deep_ep_direct_metadata=plan.use_deep_ep_direct_metadata,
            x_ready_event=ctx.x_ready_event,
        )

    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        *,
        x_sf: torch.Tensor | None = None,
        moe_output: torch.Tensor | None = None,
        enable_alltoall: bool = False,
        weight_view: NvFp4WeightView,
        recv_expert_count: torch.Tensor | None = None,
        deep_ep_expert_capacity: int | None = None,
        use_deep_ep_direct_metadata: bool = False,
        x_ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """NVFP4 MoE computation through ``run_moe_nvfp4_impl``.

        Args:
            weight_view: Bundled weight tensors.
            use_deep_ep_direct_metadata: Use adapter-free, count-native DeepEP
                metadata. The scheduler sets this only for the supported path.
            x_ready_event: Set when ``x``/``x_sf`` come from
                ``quantize_input_async``; joined inside ``run_moe_nvfp4_impl``
                right before the FC1 gather GEMM (the locality-domain path and
                empty micro-batches join here).
        """
        output_dtype = torch.bfloat16
        use_locality_domain = self._locality_domain_runtime is not None

        if (recv_expert_count is None) != (deep_ep_expert_capacity is None):
            raise ValueError(
                "recv_expert_count and deep_ep_expert_capacity must be provided together"
            )
        if use_deep_ep_direct_metadata:
            if not self.use_fused_finalize:
                raise RuntimeError(
                    "DeepEP direct metadata requires fused finalize; falling "
                    "back would sort adapter-free placeholder expert IDs"
                )
            if recv_expert_count is None or deep_ep_expert_capacity is None:
                raise RuntimeError("DeepEP direct metadata was requested without dispatch metadata")

        if moe_output is None:
            moe_output = torch.empty(
                (token_final_scales.size(0), self.hidden_size), dtype=output_dtype, device=x.device
            )
        else:
            assert moe_output.size() == (token_final_scales.size(0), self.hidden_size)
            assert moe_output.dtype == output_dtype

        # The async input quantization forked onto the aux stream must be
        # joined on every path (CUDA graph capture rejects unjoined work).
        # Only the single-tensor impl overlaps it with moe_sort.
        if use_locality_domain or token_selected_experts.size(0) == 0:
            if x_ready_event is not None:
                x_ready_event.wait()
            self._pending_x_ready_event = None
        else:
            self._pending_x_ready_event = x_ready_event

        # Empty micro-batches are valid at the backend boundary. Avoid
        # entering autotuning because its synthetic grouped-GEMM inputs
        # require at least one output row.
        if token_selected_experts.size(0) == 0:
            return moe_output

        effective_top_k = token_selected_experts.size(-1)
        use_direct_expert_metadata = use_deep_ep_direct_metadata

        if use_locality_domain:
            forward_impl = self._run_moe_nvfp4_locality_domain
            workload_identity = self._locality_domain_workload_identity(x.dtype)
            tuner_key = "CuteDslFusedMoE::run_moe_nvfp4::locality_domain_end_to_end"
            inputs = [
                x,
                token_selected_experts,
                token_final_scales,
                x_sf,
                moe_output,
            ]
        else:
            forward_impl = self.run_moe_nvfp4_impl
            workload_identity = None
            tuner_key = "CuteDslFusedMoE::run_moe_nvfp4"
            inputs = [
                x,
                token_selected_experts,
                token_final_scales,
                x_sf,
                moe_output,
                weight_view,
            ]

        tuner = AutoTuner.get()
        runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=forward_impl,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=weight_view.expert_size_per_partition,
            local_expert_offset=weight_view.slot_start,
            enable_finalize_fusion=self.use_fused_finalize,
            enable_alltoall=enable_alltoall,
            workload_identity=workload_identity,
            use_direct_expert_metadata=use_direct_expert_metadata,
            use_locality_domain=use_locality_domain,
            prime_inner_tactics=use_locality_domain,
        )

        if use_direct_expert_metadata:
            inputs.append(recv_expert_count)
        _, best_tactic = tuner.choose_one(
            tuner_key,
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def run_moe_nvfp4_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weight_view: NvFp4WeightView,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        recv_expert_count: torch.Tensor | None = None,
        deep_ep_expert_capacity: int | None = None,
        use_count_native_expert_metadata: bool = False,
    ) -> torch.Tensor:
        """NVFP4 MoE through the single-tensor CuteDSL ops."""
        output_dtype = torch.bfloat16
        sm_version = get_sm_version()
        use_rubin = sm_version == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE

        effective_top_k = token_selected_experts.size(1)
        esp = weight_view.expert_size_per_partition
        slot_start = weight_view.slot_start

        if recv_expert_count is not None:
            if deep_ep_expert_capacity is None or deep_ep_expert_capacity <= 0:
                raise ValueError(
                    "deep_ep_expert_capacity must be positive when recv_expert_count is provided"
                )
            if effective_top_k != 1:
                raise ValueError("count-native DeepEP metadata requires effective_top_k == 1")
            if recv_expert_count.dim() != 1 or recv_expert_count.numel() != esp:
                raise ValueError(
                    "recv_expert_count must be a 1D tensor with one count per "
                    f"local expert; expected {esp} elements, got shape "
                    f"{tuple(recv_expert_count.shape)}"
                )
            expected_rows = esp * deep_ep_expert_capacity
            if x.size(0) != expected_rows:
                raise ValueError(
                    "expert-major DeepEP input has the wrong row count; "
                    f"expected {expected_rows}, got {x.size(0)}"
                )
            if not use_count_native_expert_metadata:
                raise RuntimeError("Expert-major DeepEP input requires count-native metadata")
            # Count-native runners interpret each metadata argument as the
            # same expert-count tensor and do not materialize adapter arrays.
            tile_idx_to_expert_idx = recv_expert_count
            tile_idx_to_mn_limit = recv_expert_count
            expanded_idx_to_permuted_idx = recv_expert_count
            permuted_idx_to_expanded_idx = recv_expert_count
            _total_num_padded_tokens = None
            num_non_exiting_tiles = recv_expert_count
        else:
            if use_count_native_expert_metadata:
                raise RuntimeError("Count-native metadata requires DeepEP receive counts")
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

        has_aux_streams = self._has_moe_output_memset_aux_stream()
        if self.use_fused_finalize and has_aux_streams:
            memset_stream = self._moe_output_memset_run_stream()
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(memset_stream)
            with torch.cuda.stream(memset_stream):
                self.event_dict[EventType.Main].wait()
                if use_count_native_expert_metadata:
                    torch.ops.trtllm.moe_output_memset_from_expert_counts_inplace(
                        input=moe_output,
                        expert_counts=recv_expert_count,
                        expert_capacity=deep_ep_expert_capacity,
                        ep_size=self.mapping.moe_ep_size,
                        enable_alltoall=enable_alltoall,
                    )
                else:
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
                self.event_dict[EventType.MoeOutputMemset].record()

        # Fused gather + GEMM + activation + quantize for FC1.
        # For gated (SwiGLU): weights are interleaved [up, gate], output is N/2.
        # For non-gated (Relu2): weights are plain, output is N.
        gather_act_op = (
            torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin
            if use_rubin
            else torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell
        )

        gather_act_kwargs = dict(
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
        )
        if use_count_native_expert_metadata:
            gather_act_kwargs["expert_counts"] = recv_expert_count
            gather_act_kwargs["expert_capacity"] = deep_ep_expert_capacity
        if use_rubin:
            gather_act_kwargs["output_tensor"] = None
            gather_act_kwargs["output_sf_tensor"] = None
        gather_act_kwargs["activation_type"] = self.activation_type
        # ``act_alpha`` / ``act_beta`` are where ``SiTuActivation.constants()``
        # lands: gate_softcap -> alpha, linear_softcap -> beta, both reduced to
        # a uniform scalar by the shape this backend declares. Only forwarded
        # for SiTU so every other activation keeps the op's ``None`` default --
        # passing them unconditionally would make the op signature lie about
        # which kinds have soft-caps.
        if self.activation_type == ActivationType.SiTu:
            gather_act_kwargs["situ_beta"] = self.act_alpha
            gather_act_kwargs["situ_linear_beta"] = self.act_beta
        # The clamp is a SwiGLU-only constant: Relu2 layers materialize
        # act_clamp as +inf, and the kernels reject a finite limit on any
        # other activation, so only SwiGLU layers forward it.
        if self.activation_type == ActivationType.Swiglu:
            gather_act_kwargs["swiglu_limit_scalar"] = self.act_clamp

        # Join the async input quantization (quantize_input_async): x/x_sf were
        # produced on the aux stream and are first read here, after moe_sort
        # ran on the main stream in parallel with the quantization.
        self._join_pending_x_ready()
        x, x_sf = gather_act_op(**gather_act_kwargs)

        if self.use_fused_finalize:
            if has_aux_streams:
                self.event_dict[EventType.MoeOutputMemset].wait()
            elif use_count_native_expert_metadata:
                torch.ops.trtllm.moe_output_memset_from_expert_counts_inplace(
                    input=moe_output,
                    expert_counts=recv_expert_count,
                    expert_capacity=deep_ep_expert_capacity,
                    ep_size=self.mapping.moe_ep_size,
                    enable_alltoall=enable_alltoall,
                )
            else:
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

            # FC2: Grouped GEMM + Finalize (scatter-add) fusion
            finalize_inplace_op = (
                torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin
                if use_rubin
                else torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell
            )

            finalize_inplace_kwargs = dict(
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
                output_dtype=output_dtype,
            )
            if use_rubin:
                # The Rubin op has no count-native variant, so it does not
                # accept the expert-count arguments at all. Reaching here with
                # count-native metadata would mean can_use_deep_ep_direct_metadata
                # stopped excluding SM107.
                if use_count_native_expert_metadata:
                    raise NotImplementedError(
                        "Count-native DeepEP expert metadata is not supported "
                        "by the Rubin (SM107) fused-finalize grouped GEMM."
                    )
            else:
                finalize_inplace_kwargs["expert_counts"] = (
                    recv_expert_count if use_count_native_expert_metadata else None
                )
                finalize_inplace_kwargs["expert_capacity"] = (
                    deep_ep_expert_capacity if use_count_native_expert_metadata else 0
                )

            finalize_inplace_op(**finalize_inplace_kwargs)
        else:
            if use_rubin:
                # Rubin does not have a basic grouped GEMM kernel (without
                # fused finalize) yet. Force use_fused_finalize=True for Rubin.
                raise NotImplementedError(
                    "Rubin (SM107) MOE requires use_fused_finalize=True. "
                    "Basic grouped GEMM without finalize fusion is not yet "
                    "supported on Rubin."
                )
            x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
                input=x.view(torch.float4_e2m1fn_x2),
                weight=weight_view.w2_weight.view(torch.float4_e2m1fn_x2),
                input_scale=x_sf.view(torch.uint8),
                weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
                alpha=weight_view.fc2_global_scale,
                tile_idx_to_group_idx=tile_idx_to_expert_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                num_experts=self.num_slots,
                top_k=effective_top_k,
                num_local_experts=esp,
                local_expert_offset=slot_start,
                tile_size=tile_size,
                output_dtype=output_dtype,
            )
            torch.ops.trtllm.moe_unpermute_inplace(
                permuted_input=x,
                output=moe_output,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                topk_scales=token_final_scales,
            )
        return moe_output

    def _run_moe_nvfp4_locality_domain(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        x_sf: torch.Tensor | None = None,
        moe_output: torch.Tensor | None = None,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        overlap_moe_output_memset: bool = True,
    ) -> torch.Tensor:
        """locality domain path: half-weight children, shared output buffers, fork/join.

        Each child holds localized half-N weights. Both partitions use the
        same tuned tactic and write directly into their strided regions of the
        shared FC1/FC2 output buffers.
        """
        output_dtype = torch.bfloat16
        num_partitions = self._locality_domain_plan.num_partitions
        shards = self._locality_domain_weight_shards
        effective_top_k = token_selected_experts.size(-1)

        # --- moe_sort (shared, on main stream) ---
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
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            tile_tokens_dim=tile_size,
        )

        # --- Allocate shared output ---
        if moe_output is None:
            moe_output = torch.empty(
                (token_selected_experts.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device,
            )

        # --- FC1: gather + grouped GEMM + SwiGLU, fork/join ---
        # Each child has half-N weight [num_exp, inner_size, hidden_size].
        # Shared output buffers:
        #   c:    [permute_m, inner_size] — each locality domain writes half via strided layout
        #   c_sf: [full_sf_size] — each locality domain writes at K-tile offset via full_c_shape
        #   Kernel uses full_c_shape to compute sfc layout with full M-tile stride,
        #   so no copy-back or interleave is needed.
        m = permuted_idx_to_expanded_idx.size(0)
        shard_weight_n = shards[0]["w3_w1_weight"].size(1)  # half interleaved N
        shard_interm = shard_weight_n // 2  # post-SwiGLU per partition
        full_interm = shard_interm * num_partitions
        fc1_out = torch.empty(
            m, shard_interm // 2 * 2, dtype=torch.float4_e2m1fn_x2, device=x.device
        )
        full_sf_size = m * full_interm // self.scaling_vector_size
        fc1_out_sf = torch.empty(full_sf_size, dtype=torch.uint8, device=x.device)

        assert self.use_fused_finalize, (
            "locality domain MoE requires use_fused_finalize=True on Rubin"
        )

        # Inner FC tactics are prepared before outer CUDA-graph profiling. Keep
        # that first-use tuning on the main stream; normal execution still
        # overlaps this memset with the already-prepared FC1 composite op.
        memset_overlapped = overlap_moe_output_memset and self._has_moe_output_memset_aux_stream()
        if memset_overlapped:
            memset_stream = self._moe_output_memset_run_stream()
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(memset_stream)
            with torch.cuda.stream(memset_stream):
                self.event_dict[EventType.Main].wait()
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
                self.event_dict[EventType.MoeOutputMemset].record()

        torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin(
            input=x.view(torch.float4_e2m1fn_x2),
            weight_0=shards[0]["w3_w1_weight"].view(torch.float4_e2m1fn_x2),
            weight_1=shards[1]["w3_w1_weight"].view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale_0=shards[0]["fc1_weight_block"].view(torch.uint8),
            weight_scale_1=shards[1]["fc1_weight_block"].view(torch.uint8),
            alpha=shards[0]["fc1_global"],
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=shards[0]["fc2_input_scale"],
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_tensor=fc1_out,
            output_sf_tensor=fc1_out_sf,
            scaling_vector_size=self.scaling_vector_size,
            activation_type=self.activation_type,
            swiglu_limit_scalar=(
                self.act_clamp if self.activation_type == ActivationType.Swiglu else float("inf")
            ),
        )

        fc1_out_sf_merged = fc1_out_sf

        if memset_overlapped:
            self.event_dict[EventType.MoeOutputMemset].wait()
        else:
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

        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin(
            input=fc1_out.view(torch.float4_e2m1fn_x2),
            weight_0=shards[0]["w2_weight"].view(torch.float4_e2m1fn_x2),
            weight_1=shards[1]["w2_weight"].view(torch.float4_e2m1fn_x2),
            input_scale=fc1_out_sf_merged.view(torch.uint8),
            weight_scale_0=shards[0]["fc2_weight_block"].view(torch.uint8),
            weight_scale_1=shards[1]["fc2_weight_block"].view(torch.uint8),
            alpha=shards[0]["fc2_global"],
            output=moe_output,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_dtype=output_dtype,
            ep_size=self.mapping.moe_ep_size,
            enable_alltoall=enable_alltoall,
            scaling_vector_size=self.scaling_vector_size,
        )

        return moe_output

    def _locality_domain_scale_shard(
        self, partition_id: int, half_n1: int, half_n2: int
    ) -> dict[str, torch.Tensor]:
        """Split the block scales along N like the weights; share the rest."""
        fc1_cols = slice(partition_id * half_n1, (partition_id + 1) * half_n1)
        fc2_cols = slice(partition_id * half_n2, (partition_id + 1) * half_n2)
        fc1_scale_slice = self.quant_scales.fc1_weight_block[:, fc1_cols]
        fc2_scale_slice = self.quant_scales.fc2_weight_block[:, fc2_cols]
        return {
            "fc1_weight_block": _copy_to_new_cuda_allocation(fc1_scale_slice),
            "fc2_weight_block": _copy_to_new_cuda_allocation(fc2_scale_slice),
            "fc1_global": self.quant_scales.fc1_global,
            "fc2_global": self.quant_scales.fc2_global,
            "fc2_input_scale": self.fc2_input_scale,
        }
