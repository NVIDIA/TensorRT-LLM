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
"""``trtllm.triton.fp8_act_grouped_gemm.fp8_block_scales``."""

import torch

from ....utils import ActivationType
from ..activation import ActivationParamShape, MoEActivationSupport
from ..fused_moe_triton_fp8_block_scale import run_triton_fp8_block_scale_moe_fp8_act
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    check_activation_support,
    require_comm_plan,
)
from ..impl_environment import MoEEnvFlag
from ..impl_identity import register_moe_impl
from .eligibility import check_cutlass_leaf
from .fp8_block_scales_hopper import DeepgemmCudaHopperFp8BlockScalesImpl
from .identity import KERNEL_FP8_ACT_GROUPED_GEMM, TECHNIQUE_TRITON, cutlass_descriptor


def _check_opted_in(cls: type, d: MoEDeployment) -> MoEEligibility | None:
    """Decline unless the deployment opted in through ``cls.opt_in_flag``."""
    if d.env.env_flag(cls.opt_in_flag.value) == "1":
        return None
    return MoEEligibility.no(
        MoERejectReason.PATH_NOT_ENABLED,
        f"{cls.__name__} is opt-in: set {cls.opt_in_flag.value}=1 to enable it",
    )


def _check_no_expert_bias(cls: type, p: MoEProblem) -> MoEEligibility | None:
    """Decline an expert FC bias, which the Triton kernels take no operand for."""
    if p.bias:
        return MoEEligibility.no(
            MoERejectReason.ACTIVATION_UNSUPPORTED, f"{cls.__name__} takes no expert bias"
        )
    return None


@register_moe_impl
class TrtllmTritonFp8ActGroupedGemmFp8BlockScalesImpl(DeepgemmCudaHopperFp8BlockScalesImpl):
    """``trtllm.triton.fp8_act_grouped_gemm.fp8_block_scales``.

    FP8 block scales on SM90 for decode-sized batches. The grouped DeepGEMM
    kernel of the parent leaf walks every expert group on one persistent CTA
    per SM; the Triton kernels here quantize the activations per 1 x 128 group
    and launch only as many expert blocks as the batch's (token, expert) pairs
    can occupy. A call takes the Triton path when all of these hold
    (:meth:`_takes_triton_path`); otherwise it runs the parent's fused op:

    - ``run_moe`` receives at most ``TRITON_FP8_ACT_MAX_TOKENS`` rows. These
      are the rows after dispatch, so with attention DP and an allgather they
      count every DP rank's tokens.
    - The scheduler hands no ``moe_output`` buffer. The Triton path returns its
      own tensor, so it never runs under a communication strategy that
      combines from a workspace (NVLinkOneSided).
    - BF16 input without scale factors.
    - The layer's clamp, if any, is one value for every local expert
      (decided once in :meth:`cache_derived_state`).

    Opt-in: resolution passes this leaf over unless
    ``TLLM_MOE_ENABLE_SM90_TRITON_FP8_DECODE=1`` (``opt_in_flag``), so by
    default the parent leaf serves every SM90 FP8 block-scale layer.

    With expert parallelism, (token, expert) pairs owned by other ranks keep a
    clamped local expert id with a zero weight, so they are still computed.
    """

    #: Largest number of rows ``run_moe`` receives that runs the Triton path.
    TRITON_FP8_ACT_MAX_TOKENS = 32
    #: The environment flag that admits this leaf; read through ``d.env``.
    opt_in_flag = MoEEnvFlag.SM90_TRITON_FP8_DECODE

    descriptor = cutlass_descriptor(
        "fp8_block_scales",
        "Triton FP8-activation grouped GEMM for decode-sized batches, the DeepGEMM grouped "
        "GEMM otherwise, SM90.",
        technique=TECHNIQUE_TRITON,
        kernel_name=KERNEL_FP8_ACT_GROUPED_GEMM,
        # The Triton kernels take no bias operands.
        supports_expert_bias=False,
    )
    # The Triton kernels run SwiGLU only. The clamp stays a per-expert tensor
    # for the fused op; the Triton path takes it only when it is uniform.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu}),
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
        clamp_after_silu=True,
    )

    #: ``(limit, clamp_after_silu)`` for the Triton SwiGLU, or None while the
    #: layer cannot take the Triton path. Set by :meth:`cache_derived_state`.
    _triton_swiglu: tuple[float | None, bool] | None = None

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(
            cls,
            p,
            d,
            check_activation_support(cls, p),
            _check_no_expert_bias(cls, p),
            _check_opted_in(cls, d),
        )

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        self._triton_swiglu = self._uniform_swiglu_clamp()

    def _uniform_swiglu_clamp(self) -> tuple[float | None, bool] | None:
        """The Triton SwiGLU settings read off the installed ``act_*`` slots.

        Returns ``(limit, clamp_after_silu)``, or None when the layer has an
        expert bias, alpha/beta constants, or a clamp that differs between
        experts: the Triton kernel takes the clamp by value.
        """
        if (
            self.activation_type != ActivationType.Swiglu
            or self.w3_w1_bias is not None
            or self.w2_bias is not None
            or self.act_alpha is not None
            or self.act_beta is not None
        ):
            return None
        clamp = self.act_clamp
        if clamp is None:
            return None, False
        if isinstance(clamp, torch.Tensor):
            flat = clamp.detach().reshape(-1)
            if flat.numel() == 0 or not bool((flat == flat[0]).all()):
                return None
            clamp = flat[0].item()
        return float(clamp), bool(self.act_clamp_after_silu)

    def _takes_triton_path(self, ctx: MoERunContext) -> bool:
        """Whether this call runs the Triton kernels; see the class docstring."""
        return (
            self._triton_swiglu is not None
            and ctx.x.shape[0] <= self.TRITON_FP8_ACT_MAX_TOKENS
            and ctx.x.dtype == torch.bfloat16
            and ctx.x_sf is None
            and require_comm_plan(self, ctx).moe_output is None
        )

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        if not self._takes_triton_path(ctx):
            return super().run_moe(ctx, workspace=workspace)
        del workspace  # The Triton path allocates its own intermediates.

        # The scheduler leaves token_final_scales as None when the router
        # weights were already folded into x.
        token_final_scales = ctx.token_final_scales
        if token_final_scales is None:
            token_final_scales = torch.ones_like(ctx.token_selected_experts, dtype=torch.float32)
        if self.ep_size == 1:
            # Every routed slot is local: the IDs and weights need no remap.
            local_ids, local_scales = ctx.token_selected_experts, token_final_scales
        else:
            # The kernel indexes local experts; zero the scale of every pair
            # this rank does not own.
            local_ids, offsets = self._local_slot_ids(ctx.token_selected_experts)
            local_scales = token_final_scales * (local_ids == offsets).to(token_final_scales.dtype)

        swiglu_limit, swiglu_clamp_after_silu = self._triton_swiglu
        return run_triton_fp8_block_scale_moe_fp8_act(
            ctx.x,
            local_ids,
            local_scales,
            self.w3_w1_weight,
            self.quant_scales.fc_weight_scales,
            self.w2_weight,
            self.quant_scales.proj_weight_scales,
            swiglu_limit=swiglu_limit,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
            output_dtype=ctx.output_dtype,
        )
