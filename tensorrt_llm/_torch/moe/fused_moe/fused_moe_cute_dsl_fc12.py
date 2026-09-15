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

Drives ``cute_dsl_nvfp4_fc12_fused_rubin``: one persistent kernel does gather +
FC1 GEMM + gated epilogue + requant + FC2 GEMM + finalize. The epilogue is a
trace-time specialization of SwiGLU or SiTU (Kimi K3); Relu2 is not gated and
is not supported. The locality-domain path is not enabled for this backend.

``quantize_input``, ``run_moe``, the weight lifecycle, capabilities, and EPLB
are inherited from :class:`CuteDslFusedMoE`. The SiTU constants arrive in the
generic activation slots ``act_alpha`` (gate soft-cap) and ``act_beta`` (linear
soft-cap), written by ``_write_activation_slot`` from this class's own
``activation_support``; they are forwarded into the fused op and are part of the
compiled-kernel cache key. ``activation_support`` must be declared here rather
than inherited: the parent executes SwiGLU/Relu2, this backend SwiGLU/SiTU, and
``_reject_unsupported_activation`` reads the class declaration.

.. note::
   The ``trtllm::cute_dsl_nvfp4_fc12_fused_rubin`` custom op that this backend
   drives is not registered yet -- the kernel lands here first and the op
   wrapper follows. :meth:`CuteDslFc12FusedMoE.can_implement` therefore rejects
   the backend whenever the op is absent, so resolution falls through to the
   next candidate instead of failing at dispatch. The gate becomes a no-op once
   the wrapper is registered.
"""

from typing import Optional

import torch

from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...autotuner import AutoTuner
from ...cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from .activation import ActivationParamShape, MoEActivationSupport
from .fused_moe_cute_dsl import CuteDslFusedMoE, CuteDslFusedMoENvfp4Runner, NvFp4WeightView
from .impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERejectReason
from .interface import _reject


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


class CuteDslFc12FusedMoE(CuteDslFusedMoE):
    """FC1+FC2-fused CuteDSL NVFP4 MoE backend (Rubin/SM107).

    SwiGLU and SiTU share the gated FC1 geometry; the fused kernel specializes
    the epilogue at trace time. The locality-domain path is not enabled for this backend.
    """

    # Narrower and wider than the parent's: no Relu2 (the fused FC12 epilogue is
    # gated-only), plus SiTU, whose two soft-caps ride the alpha/beta slots as
    # baked scalars. Mirrors MegaMoEDeepGemm, the other SwiGLU/SiTU backend.
    # ``limit_when_absent`` stays inf: the op takes ``swiglu_limit`` by value and
    # has no "clamp absent" encoding.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.SiTu}),
        alpha_beta=ActivationParamShape.UNIFORM_SCALAR,
        limit=ActivationParamShape.UNIFORM_SCALAR,
        limit_when_absent=float("inf"),
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """FC12 fused CuteDSL: NVFP4 on Rubin (SM107) only.

        Reads only ``p``, ``d`` and process-static build capabilities (no
        ``get_sm_version()``, no ``os.environ``); the frozen environment lives
        in ``d.env``. Narrower than the parent (SM107 + NVFP4 only) because the
        FC12 fused kernel targets Rubin NVFP4; broaden when other shapes are
        ported.
        """
        sm_version = d.env.sm

        # Output is hardcoded to bfloat16 (inherited), so activation must match.
        if p.dtype_act != torch.bfloat16:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"CuteDslFc12FusedMoE only supports bfloat16 activation, got {p.dtype_act}",
            )

        if p.activation_type not in (ActivationType.Swiglu, ActivationType.SiTu):
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"CuteDslFc12FusedMoE supports only SwiGLU and SiTU, got {p.activation_type.name}",
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

        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            return _reject(
                MoERejectReason.DEP_MISSING,
                "CuteDslFc12FusedMoE (SM107 NVFP4) requires CuTE DSL internal",
            )

        # The fused op always scatter-adds into ``moe_output``; there is no
        # unfused FC2 seam to hand back. ``can_implement`` does not chain to the
        # parent, so this repeats the parent's SM107 guard rather than inheriting
        # it (``fused_moe_cute_dsl.py``, FINALIZE_FUSION_REQUIRED).
        if not d.fused_finalize_enabled:
            return _reject(
                MoERejectReason.FINALIZE_FUSION_REQUIRED,
                "CuteDslFc12FusedMoE always fuses finalize and has no unfused FC2 path",
            )

        # See the module docstring: the fused op wrapper is not registered yet.
        # Reject here rather than raising from run_moe_nvfp4_impl, so a build
        # without the op simply resolves to another backend.
        if not hasattr(torch.ops.trtllm, "cute_dsl_nvfp4_fc12_fused_rubin"):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "CuteDslFc12FusedMoE requires the trtllm::cute_dsl_nvfp4_fc12_fused_rubin "
                "custom op, which is not registered in this build",
            )

        return MoEEligibility.ok()

    # ------------------------------------------------------------------
    # Fused-kernel dispatch. Mirrors the parent's non-locality-domain NVFP4 path but
    # (1) uses a distinct autotuner key + outer runner so FC12 tactics do
    # not collide with the parent CuteDSL backend, and (2) drives the fused
    # single-op path in ``run_moe_nvfp4_impl``. Locality domains are never used here.
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
            # Inner kernel_cache already keys on these; the outer AutoTuner
            # unique_id must too, or SwiGLU/SiTU share a tile-size tactic.
            workload_identity=(
                int(self.activation_type),
                self.act_alpha,
                self.act_beta,
            ),
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
        overlap_moe_output_memset: bool = True,
    ) -> torch.Tensor:
        """Single fused FC1+FC2 op (replaces the parent's two-op sequence).

        ``overlap_moe_output_memset`` is accepted and ignored: the inherited
        ``CuteDslFusedMoENvfp4Runner.forward`` passes it when priming the tile
        caches, but the fused op issues the ``moe_output`` memset internally so
        the caller cannot sequence it. Priming output is discarded, so ignoring
        it is safe.
        """
        del overlap_moe_output_memset
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

        # One fused op: gather + FC1 GEMM + gated act (SwiGLU or SiTU) + requant
        # + FC2 GEMM + finalize (scatter-add into moe_output).
        # fc1_alpha/fc2_alpha map 1:1 to the two-op path's per-expert global
        # scales (the kernel takes split alphas). The three atomic counters are
        # allocated + memset inside the op runner. The moe_output zeroing memset
        # is issued inside the op too (right before the fused kernel, so it is
        # the kernel's immediate stream predecessor and the PDL prologue can
        # overlap it); expanded_idx_to_permuted_idx / ep_size / enable_alltoall
        # are passed through for that memset.
        situ = ActivationType(self.activation_type) == ActivationType.SiTu
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
            activation_type=int(self.activation_type),
            situ_beta=self.act_alpha if situ else -1.0,
            situ_linear_beta=self.act_beta if situ else -1.0,
        )
        return moe_output
