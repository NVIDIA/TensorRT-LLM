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
"""CuteDslFc12FusedMoE: FC1+FC2-fused CuteDSL NVFP4 MoE backend for Rubin (SM107).

SKELETON STATE
--------------
This is a thin subclass of :class:`CuteDslFusedMoE` that currently REUSES the
existing CuteDSL FC1/FC2 ops and kernels, so the backend is fully functional and
end-to-end testable today (it behaves like ``CuteDslFusedMoE`` but is a distinct,
separately selectable backend ``"CUTEDSL_FC12"``).

When the colleague's fused FC12 kernel (FC1+FC2 fused into one persistent kernel)
is delivered, only the marked swap points need changing -- everything else
(``quantize_input``, ``run_moe``, weight lifecycle, EPLB) is inherited from
:class:`CuteDslFusedMoE`:

  1. ``run_moe_nvfp4_impl``  -- replace the parent's FC1+FC2 two-op sequence with
     the single fused op ``cute_dsl_nvfp4_fc12_fused_rubin``.
  2. ``_get_quant_method``   -- override ONLY if the fused kernel's weight/scale
     layout differs from ``NVFP4CuteDslFusedMoEMethod`` (then also re-evaluate
     ``eplb_support_status`` for the new method).

uGPU is intentionally NOT wired up for this backend: benchmarks currently run
without the uGPU feature, so FC12 is a non-uGPU backend for now. It does not
override ``_run_moe_nvfp4_ugpu`` and the ``plan_moe`` gate in ``ugpu/policy.py``
does not enable uGPU for ``CUTEDSL_FC12``. uGPU can be added back later if needed.

See ``CuteDslFc12FusedMoE-integration-plan.txt`` for the full plan.
"""

from typing import Optional

import torch

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...autotuner import AutoTuner
from ...cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from .fused_moe_cute_dsl import (CuteDslFusedMoE, CuteDslFusedMoENvfp4Runner,
                                 NvFp4WeightView)
from .impl_contract import (MoEDeployment, MoEEligibility, MoEProblem,
                            MoERejectReason)
from .interface import _reject


class CuteDslFc12FusedMoENvfp4Runner(CuteDslFusedMoENvfp4Runner):
    """Outer autotune runner for the fused FC12 backend.

    Identical to the parent except the routing-tile candidate set: the v1
    fused kernel only supports the 128-wide tile, so restrict the valid
    tactics to ``[128]`` (the parent offers 128/256, which would make the
    inner fused runner assert on 256 during autotuning).
    """

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return [128]


class CuteDslFc12FusedMoE(CuteDslFusedMoE):
    """FC1+FC2-fused CuteDSL NVFP4 MoE backend (Rubin/SM107).

    Currently a functional skeleton that reuses ``CuteDslFusedMoE``'s ops and
    kernels; the FC12 fused kernel is swapped in at the marked point once
    delivered. ``scheduler_kind`` (EXTERNAL_COMM), ``quantize_input``,
    ``run_moe``, the weight lifecycle, capabilities, and EPLB are inherited
    unchanged. uGPU is not enabled for this backend (benchmarks run without it).
    """

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """FC12 fused CuteDSL: NVFP4 on Rubin (SM107) only.

        Pure gate -- reads only ``p`` and ``d`` (no ``get_sm_version()``, no
        import probes, no ``os.environ``); the frozen environment lives in
        ``d.env``. Narrower than the parent (SM107 + NVFP4 only) because the FC12
        fused kernel targets Rubin NVFP4; broaden when other shapes are ported.
        """
        sm_version = d.env.sm

        # Output is hardcoded to bfloat16 (inherited), so activation must match.
        if p.dtype_act != torch.bfloat16:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                "CuteDslFc12FusedMoE only supports bfloat16 activation, "
                f"got {p.dtype_act}")

        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "CuteDslFc12FusedMoE does not support swiglu_gptoss_style")

        if p.quant_algo != QuantAlgo.NVFP4:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                "CuteDslFc12FusedMoE only supports NVFP4, "
                f"got quant_algo={p.quant_algo}")

        if sm_version != 107:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                "CuteDslFc12FusedMoE targets Rubin (SM107), "
                f"got SM{sm_version}")

        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            return _reject(
                MoERejectReason.DEP_MISSING,
                "CuteDslFc12FusedMoE (SM107 NVFP4) requires CuTe DSL Rubin support"
            )

        return MoEEligibility.ok()

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
                (token_final_scales.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device)
        else:
            assert moe_output.size() == (token_final_scales.size(0),
                                         self.hidden_size)
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
        inputs = [
            x, token_selected_experts, token_final_scales, x_sf, moe_output,
            weight_view
        ]
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

        (tile_idx_to_expert_idx, tile_idx_to_mn_limit,
         expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
         total_num_padded_tokens,
         num_non_exiting_tiles) = torch.ops.trtllm.moe_sort(
             token_selected_experts=token_selected_experts,
             token_final_scales=token_final_scales,
             num_experts=self.num_slots,
             top_k=effective_top_k,
             local_expert_offset=slot_start,
             local_num_experts=esp,
             tile_tokens_dim=tile_size,
         )

        # Zero moe_output for the scatter-add finalize via the sparse memset;
        # faster than a full-buffer output zero_() at large token counts (the
        # fused op's finalize scatter-ADDs into moe_output, so it must start
        # zeroed). Replaces the in-kernel fc2_c.zero_().
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

        # One fused op: gather + FC1 GEMM + SwiGLU + requant + FC2 GEMM +
        # finalize (scatter-add into moe_output = a2a combine workspace).
        # fc1_alpha/fc2_alpha map 1:1 to the two-op path's per-expert global
        # scales (the kernel takes split alphas). The three atomic counters are
        # allocated + memset inside the op runner.
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
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
            scaling_vector_size=16,
            swiglu_limit=self.act_clamp,
        )
        return moe_output
