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
"""DeepSeek-style FP8 with 1x128 block scales, served on both providers.

``run_fp8_block_scale_moe`` is its own kernel ABI, and this format is also the
only one that narrows the family's activation support and the only one whose
grouped GEMM can absorb the shared experts. None of that is shared with
:mod:`.fp4_block_scale`, so the two sit side by side rather than under a common
quant layer.
"""

from dataclasses import replace

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

from ..activation import ActivationParamShape, MoEActivationSupport
from ..impl_contract import MoERunContext
from ..impl_environment import MoEEnvFlag, collect_moe_environment
from ..quantization import DeepSeekFP8BlockScalesFusedMoEMethod
from .base import TrtllmGenFusedMoEBase
from .kernel_inputs import prepare_kernel_inputs


class TRTLLMGenFp8BlockScalesBase(TrtllmGenFusedMoEBase):
    """DeepSeek-style FP8 with 1x128 block scales.

    The only format whose activation is quantized inside ``run_moe`` rather
    than in ``quantize_input``: ``fp8_quantize_1x128`` returns scales shaped
    ``(blocked_n, num_tokens)``, and the all-to-all dispatch needs every
    payload's first dimension to be ``num_tokens``. Transposing around the
    dispatch would cost more than it saves, so this format simply does not
    offer post-quant communication.

    Also the only format that departs from the family's activation ABI, and
    the only one whose grouped GEMM can absorb the shared experts.
    """

    def resolve_activation_support(self) -> MoEActivationSupport:
        """Narrow the clamp ABI to the scalar this format's kernel reads.

        The DeepSeek FP8 block-scale path runs the clamp in a separate
        activation kernel (``DevKernel.cu::activationDeepSeekKernel``) taking
        one ``float`` by value, so the per-expert tensor the family declares
        would be silently ignored. Narrowed from ``type(self)`` so the clamp is
        the only thing decided here and a leaf's own ``activation_support``
        survives.
        """
        return replace(
            type(self).activation_support,
            limit=ActivationParamShape.UNIFORM_SCALAR,
        )

    @classmethod
    def fused_shared_expert_count(cls, model_config: ModelConfig) -> int:
        """Fold the shared experts into the routed grouped GEMM, if asked to.

        Opt-in: set ``TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1``. The benefit is
        workload-dependent (small decode batches gain, large prefill chunks lose
        the aux-stream overlap of the unfused path), and the fused path
        additionally restricts tactics to tileN>=32 to avoid a small-tile dynB
        kernel defect.

        The flag is read through the frozen selection environment rather than
        ``os.environ``, so it lands in the fingerprint alongside the other
        choices that decide which leaf is eligible.
        """
        if collect_moe_environment().env_flag(MoEEnvFlag.SHARED_EXPERT_FUSION) != "1":
            return 0
        # Only the trtllm op backend implements fused shared experts, so the
        # FlashInfer leaf on this same format stays unfused.
        if cls.use_flashinfer:
            return 0
        # Expert parallelism (moe_ep_size > 1) is not supported by the fused
        # path yet (the routing kernel's shared-expert append assumes the full
        # expert set is local); gate it out here so EP configs fall back to the
        # unfused path instead of tripping the runtime EP check in the
        # TRTLLM-Gen runner.
        if model_config.mapping.dp_size != 1 or model_config.mapping.moe_ep_size != 1:
            return 0
        # Not all models that use this backend define shared experts (e.g.
        # non-DeepSeek MoEs), so fall back to 0 when the config has no
        # `n_shared_experts`.
        return getattr(model_config.pretrained_config, "n_shared_experts", 0) or 0

    def _create_quant_method_weights(self) -> None:
        """This format's method sizes the expert dimension for the fused slots."""
        self.quant_method.create_weights(self, self.num_fused_shared_expert)

    def fuse_shared_expert(self, shared_experts: GatedMLP) -> None:
        assert self._weights_created
        self.quant_method.fuse_shared_expert(self, shared_experts, self.num_fused_shared_expert)

    def _check_configs(self) -> None:
        """No FC bias and no SwiGLU constants: this format has no fused cubin.

        Narrower than ``supports_gptoss_style`` already covers, because that
        gate only sees the gpt-oss package as a whole; a checkpoint can carry a
        plain expert bias without it.
        """
        assert not self.bias and self.act_alpha is None and self.act_beta is None, (
            f"{type(self).__name__} takes no expert bias and no swiglu alpha/beta constants."
        )

    def _get_quant_method(self) -> object:
        return DeepSeekFP8BlockScalesFusedMoEMethod()

    def quantize_input(
        self, x: torch.Tensor, post_quant_comm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor | tuple:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = prepare_kernel_inputs(self, ctx)

        assert k.do_finalize, "fp8_block_scale_moe_runner does not support do_finalize=False"
        x, x_sf = k.x, k.x_sf
        # fp8_quantize_1x128 returns 2D x_sf on SM100+, 1D on SM90
        if x_sf is None:
            x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)

        result = self.op_backend.run_fp8_block_scale_moe(
            k.router_logits,
            k.routing_bias,
            x,
            x_sf,
            self.w3_w1_weight,
            self.w3_w1_weight_scaling_factor,
            self.w2_weight,
            self.w2_weight_scaling_factor,
            self.num_slots,
            k.top_k,
            self.num_fused_shared_expert,
            k.n_group,
            k.topk_group,
            self.intermediate_size_per_partition,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            gemm1_clamp_limit=self.act_clamp,
            output=k.moe_output,
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )
        # When output is provided, use it directly as the result
        return k.moe_output if k.moe_output is not None else result
