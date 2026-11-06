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
"""``TrtllmCutedslGroupedGemmBf16Impl``: ``trtllm.cutedsl.grouped_gemm.none``."""

import torch

from tensorrt_llm._utils import get_sm_version

from .....autotuner import AutoTuner
from .....cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from .....utils import ActivationType, EventType, Fp4QuantizedTensor
from ...activation import MoEActivationSupport
from ...cutlass.eligibility import SmSupport
from ...impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERunContext,
    MoEStaticCapability,
    check_activation_support,
    require_comm_plan,
)
from ...impl_identity import MoEImplDescriptor, register_moe_impl
from ...interface import MoESchedulerKind
from ...quantization import BF16CuteDslFusedMoEMethod
from ..identity import grouped_gemm_identity
from ..kernel_inputs import finalize_routing_scales
from ..runners import CuteDslFusedMoEBF16Runner
from .base import CuteDslFusedMoEBase
from .eligibility import check_grouped_gemm_leaf


@register_moe_impl
class TrtllmCutedslGroupedGemmBf16Impl(CuteDslFusedMoEBase):
    """``trtllm.cutedsl.grouped_gemm.none``: CuteDSL BF16 grouped GEMM."""

    descriptor = MoEImplDescriptor(
        identity=grouped_gemm_identity("none"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=MoEStaticCapability(
            supports_eplb=True, supports_apply_router_weight_on_input=True
        ),
        input_requirement=MoEInputRequirement(routing_scales_dtype=torch.float32),
        doc="CuTe DSL BF16 grouped GEMM on Rubin (SM107): gather + FC1 + "
        "fused SwiGLU, then FC2 + fused finalize.",
    )
    sm_support = SmSupport(allowed=frozenset({107}))

    # The BF16 FC1 op fuses a plain SwiGLU (the BF16 method interleaves FC1
    # weights for it) and takes no activation constants, so no clamp.
    activation_support = MoEActivationSupport(kinds=frozenset({ActivationType.Swiglu}))

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_grouped_gemm_leaf(cls, p, d, check_activation_support(cls, p))

    def _get_quant_method(self) -> object:
        # Refuse a direct construction that selection would not make, before it
        # allocates weights for kernels this machine cannot run.
        sm = get_sm_version()
        if not self.sm_support.accepts(sm):
            raise ValueError(
                f"{type(self).__name__} supports {self.sm_support.describe()}, got SM{sm}"
            )
        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            raise ValueError(f"{type(self).__name__} requires Rubin support in CuTe DSL")
        return BF16CuteDslFusedMoEMethod()

    def quantize_input(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        post_quant_comm: bool = True,
    ) -> tuple[torch.Tensor, None]:
        """BF16 activations go to the kernels as they are; no scale factors."""
        del post_quant_comm
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        """Run the BF16 MoE computation; returns final_hidden_states."""
        del workspace  # CuteDSL kernels allocate their own intermediates.
        plan = require_comm_plan(self, ctx)
        return self.run_moe_bf16(
            x=ctx.x,
            token_selected_experts=ctx.token_selected_experts,
            token_final_scales=finalize_routing_scales(self, ctx),
            moe_output=plan.moe_output,
            enable_alltoall=plan.enable_alltoall,
        )

    def run_moe_bf16(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        moe_output: torch.Tensor | None = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        """Autotuner wrapper for BF16 MoE on Rubin (SM107)."""
        # The FC2 op always fuses finalize; can_implement rejects deployments
        # that disable it.
        assert self.use_fused_finalize, "BF16 CuteDSL MoE has no unfused FC2 path"
        output_dtype = x.dtype
        effective_top_k = token_selected_experts.size(-1)

        if moe_output is None:
            moe_output = torch.empty(
                (token_selected_experts.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device,
            )
        else:
            assert moe_output.size() == (token_selected_experts.size(0), self.hidden_size)
            assert moe_output.dtype == output_dtype

        if token_selected_experts.size(0) == 0:
            return moe_output

        self._ensure_bf16_alpha(x.device)

        use_locality_domain = self._locality_domain_runtime is not None
        if use_locality_domain:
            forward_impl = self._run_moe_bf16_locality_domain
            workload_identity = self._locality_domain_workload_identity(x.dtype)
            tuner_key = "CuteDslFusedMoE::run_moe_bf16::locality_domain_end_to_end"
        else:
            forward_impl = self.run_moe_bf16_impl
            workload_identity = None
            tuner_key = "CuteDslFusedMoE::run_moe_bf16"

        tuner = AutoTuner.get()
        runner = CuteDslFusedMoEBF16Runner(
            forward_impl=forward_impl,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            enable_alltoall=enable_alltoall,
            output_dtype=output_dtype,
            workload_identity=workload_identity,
            prime_inner_tactics=use_locality_domain,
        )

        inputs = [x, token_selected_experts, token_final_scales, moe_output]
        _, best_tactic = tuner.choose_one(
            tuner_key,
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def _ensure_bf16_alpha(self, device: torch.device) -> torch.Tensor:
        if (
            not hasattr(self, "_bf16_alpha")
            or self._bf16_alpha is None
            or self._bf16_alpha.device != device
            or self._bf16_alpha.size(0) != self.expert_size_per_partition
        ):
            self._bf16_alpha = torch.ones(
                self.expert_size_per_partition, dtype=torch.float32, device=device
            )
        return self._bf16_alpha

    def run_moe_bf16_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        moe_output: torch.Tensor,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        """BF16 MoE implementation using CuTE DSL Rubin kernels.

        FC1: gather + grouped GEMM + SwiGLU fusion
        FC2: grouped GEMM + finalize (scatter-add) fusion
        """
        output_dtype = x.dtype
        effective_top_k = token_selected_experts.size(-1)

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

        # Overlap the moe_output memset with FC1.
        has_aux = self._has_moe_output_memset_aux_stream()
        if has_aux:
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(self._moe_output_memset_run_stream())

        # Unit per-expert alpha; run_moe_bf16 allocates it before CUDA graph
        # capture.
        self._ensure_bf16_alpha(x.device)

        fc1_out = torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin(
            input=x,
            weight=self.w3_w1_weight,
            alpha=self._bf16_alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_tensor=None,
            partition_id=-1,
        )

        # Zero the moe_output rows the finalize kernel does not write.
        if has_aux:
            with torch.cuda.stream(self._moe_output_memset_run_stream()):
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

        torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin(
            input=fc1_out,
            weight=self.w2_weight,
            output=moe_output,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_dtype=output_dtype,
        )
        return moe_output

    def _run_moe_bf16_locality_domain(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        moe_output: torch.Tensor | None = None,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        overlap_moe_output_memset: bool = True,
    ) -> torch.Tensor:
        """locality domain path for unquantized BF16 MoE on Rubin."""
        output_dtype = x.dtype
        num_partitions = self._locality_domain_plan.num_partitions
        shards = self._locality_domain_weight_shards
        effective_top_k = token_selected_experts.size(-1)

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

        if moe_output is None:
            moe_output = torch.empty(
                (token_selected_experts.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device,
            )
        else:
            assert moe_output.size() == (token_selected_experts.size(0), self.hidden_size)
            assert moe_output.dtype == output_dtype

        self._ensure_bf16_alpha(x.device)

        m = permuted_idx_to_expanded_idx.size(0)
        shard_weight_n = shards[0]["w3_w1_weight"].size(1)
        shard_interm = shard_weight_n // 2
        full_interm = shard_interm * num_partitions
        fc1_out = torch.empty(m, full_interm, dtype=output_dtype, device=x.device)

        assert self.use_fused_finalize, (
            "locality domain MoE requires use_fused_finalize=True on Rubin"
        )

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

        torch.ops.trtllm.cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin(
            input=x,
            weight_0=shards[0]["w3_w1_weight"],
            weight_1=shards[1]["w3_w1_weight"],
            alpha=self._bf16_alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_tensor=fc1_out,
        )

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

        torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin(
            input=fc1_out,
            weight_0=shards[0]["w2_weight"],
            weight_1=shards[1]["w2_weight"],
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
        )

        return moe_output
