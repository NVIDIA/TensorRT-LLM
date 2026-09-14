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
"""``run_fp4_block_scale_moe`` and the three formats that reach it.

:class:`.TRTLLMGenFp4BlockScaleBase` owns the kernel call; NVFP4, W4A16_MXFP4
and W4A8_MXFP4_MXFP8 differ in how weights and inputs are prepared and not at
all in how the kernel is called, so each subclass supplies only the
preparation. Six of the eleven leaves land here, two per format.

A format's class also answers whatever the family base would otherwise switch
on ``quant_config`` for: the SiTu weight alignment and scale group size are
per-format, so they are settled here rather than branched on above.
"""

from typing import Optional, Union

import torch
from torch import nn

from ....utils import ActType_TrtllmGen, Fp4QuantizedTensor, MxFp8QuantizedTensor
from ..activation import materialize_activation_params, resolve_activation_support
from ..impl_contract import MoERunContext
from ..quantization import (
    NVFP4TRTLLMGenFusedMoEBaseMethod,
    NVFP4TRTLLMGenFusedMoEMethod,
    W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
    W4A16MXFP4TRTLLMGenFusedMoEMethod,
)
from .base import TrtllmGenFusedMoEBase
from .eligibility import nvfp4_needs_padded_method
from .kernel_inputs import get_data_or_none, prepare_kernel_inputs, to_trtllm_gen_act_type


class TRTLLMGenFp4BlockScaleBase(TrtllmGenFusedMoEBase):
    """``run_fp4_block_scale_moe`` for the three formats that share it.

    NVFP4, W4A16_MXFP4 and W4A8_MXFP4_MXFP8 differ in how weights and inputs
    are prepared and not at all in how the kernel is called, so the call lives
    here once and each subclass supplies the preparation. Six of the eleven
    leaves reach the kernel through this body.
    """

    #: Group size the SiTu cubins were built for. A property of the format,
    #: unlike ``supports_situ``; ``None`` where there are no such cubins.
    situ_scaling_vector_size: Optional[int] = None

    def _create_quant_method_weights(self) -> None:
        """Also promote SiTu's soft-caps to parameter slots the cubin indexes.

        SiTu's two soft-caps ride in the same ``gemm1_alpha`` / ``gemm1_beta``
        op slots SwiGLU's alpha/beta use; the kinds are mutually exclusive.
        They are backend configuration rather than checkpoint weights, which is
        why ``cache_derived_state`` below has to refill them. Runs before
        ``_check_configs``, which reads the slots back.
        """
        super()._create_quant_method_weights()
        if self.is_situ_activation:
            self.act_alpha = nn.Parameter(self.act_alpha, requires_grad=False)
            self.act_beta = nn.Parameter(self.act_beta, requires_grad=False)

    def _check_configs(self) -> None:
        """SiTu invariants that only hold once the weights exist.

        Everything about SiTu that is a function of the problem is a gate in
        :mod:`.eligibility`; what is left needs the allocated tensors, so it
        cannot be answered before ``create_weights``.
        """
        if not self.is_situ_activation:
            return
        if self.scaling_vector_size != self.situ_scaling_vector_size:
            raise ValueError(
                "TRTLLM-Gen SiTu requires scaling vector size "
                f"{self.situ_scaling_vector_size} for this quantization mode, "
                f"got {self.scaling_vector_size}."
            )
        # For SiTu these hold the backend-local activation parameters
        # (populated by create_weights, which runs before this check).
        for name in ("act_alpha", "act_beta"):
            value = getattr(self, name)
            if (
                value.dtype != torch.float32
                or value.shape != (self.expert_size_per_partition,)
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous float32 tensor with "
                    "one value per local expert/slot."
                )

    def cache_derived_state(self) -> None:
        """Refill the SiTu soft-caps after meta-device materialization.

        The model loader calls this on every module once storage is bound,
        because materialization wipes anything that is not a checkpoint tensor,
        and the soft-caps live in ``nn.Parameter`` slots despite being backend
        configuration. Rebuilt from ``self.activation`` and not from a snapshot
        taken in ``create_weights``: under meta init those slots are themselves
        meta at that point, so a snapshot would carry no values.
        """
        super().cache_derived_state()
        if not self.is_situ_activation:
            return
        params = materialize_activation_params(
            self.activation,
            resolve_activation_support(self),
            num_local_experts=self.expert_size_per_partition,
            device=self.act_alpha.device,
            owner=type(self).__name__,
        )
        self.act_alpha.data.copy_(params.alpha)
        self.act_beta.data.copy_(params.beta)

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = prepare_kernel_inputs(self, ctx)

        act_type = to_trtllm_gen_act_type(self.activation_type)
        factor = 1 if act_type in [ActType_TrtllmGen.Relu2, ActType_TrtllmGen.Silu] else 2
        intermediate_size_per_partition_padded = self.w3_w1_weight.shape[-2] // factor
        # Holds SwiGLU's per-expert alpha/beta, or SiTu's backend-local
        # activation parameters (which reuse this storage; see create_weights).
        gemm1_alpha, gemm1_beta = self.act_alpha, self.act_beta

        output1_scale_scalar = get_data_or_none(self, "fc31_scale_c")
        output1_scale_gate_scalar = get_data_or_none(self, "fc31_alpha")
        output2_scale_scalar = get_data_or_none(self, "fc2_alpha")

        outputs = self.op_backend.run_fp4_block_scale_moe(
            k.router_logits,
            k.routing_bias,
            k.x,
            k.x_sf,
            self.w3_w1_weight,
            self.w3_w1_weight_scale,
            self.w3_w1_bias if self.bias else None,
            gemm1_alpha,
            gemm1_beta,
            self.act_clamp,
            self.w2_weight,
            self.w2_weight_scale,
            self.w2_bias if self.bias else None,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            self.num_slots,
            k.top_k,
            k.n_group,
            k.topk_group,
            intermediate_size_per_partition_padded,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            do_finalize=k.do_finalize,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            valid_hidden_size=self.hidden_size,
            valid_intermediate_size=getattr(
                self.quant_method, "intermediate_size_per_partition_lean", None
            ),
            gated_act_type=act_type,
            output=k.moe_output,
            # Pass that to the autotuner so the top bucket profiles per-expert load at runtime scale.
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )

        if not k.do_finalize:
            return self._unfinalized(outputs)

        # When output is provided, use it directly as the result
        final_hidden_states = k.moe_output if k.moe_output is not None else outputs
        # Slice output if it was padded (only needed when moe_output is not provided)
        if k.moe_output is None and final_hidden_states.shape[1] > self.hidden_size:
            final_hidden_states = final_hidden_states[:, : self.hidden_size].contiguous()
        return final_hidden_states


class TRTLLMGenNvfp4Base(TRTLLMGenFp4BlockScaleBase):
    """NVFP4 weights and activations, group-16 block scales."""

    supports_gptoss_style = True
    # Group size of the ``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*`` cubins. Whether a
    # leaf may reach them is ``supports_situ``, declared per leaf because only
    # the native op backend calls them.
    situ_scaling_vector_size = 16

    def _situ_tp_weight_alignment(self) -> int:
        """Whole group-16 scale groups per rank, not the storage alignment.

        The resolved 32/128/256 is where the loader pads each logical shard to,
        which it does per shard after slicing. What TP has to preserve is the
        scaling vector.
        """
        return NVFP4TRTLLMGenFusedMoEMethod.scaling_vector_size

    def _get_quant_method(self):
        # SiTu fills the act_alpha/act_beta slots from create_weights, which
        # runs after this, so keying off the tensor would make the method
        # depend on *when* it is asked for.
        needs_padded_method = nvfp4_needs_padded_method(
            self.activation_type, self.act_alpha is not None
        )
        return (
            NVFP4TRTLLMGenFusedMoEMethod()
            if needs_padded_method
            else NVFP4TRTLLMGenFusedMoEBaseMethod()
        )

    def quantize_input(self, x, post_quant_comm: bool = True):
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, (
                "Fp4QuantizedTensor should not be swizzled before communication"
            )
            x_row = x.shape[0]
            x, x_sf = x.fp4_tensor, x.scaling_factor
        elif isinstance(x, MxFp8QuantizedTensor):
            assert not x.is_sf_swizzled, (
                "MxFp8QuantizedTensor should not be swizzled before communication"
            )
            x_row = x.shape[0]
            x, x_sf = x.fp8_tensor, x.scaling_factor
        else:
            # Apply pre_quant_scale if it exists (for NVFP4_AWQ)
            # fc31_act_scale shape: (1, hidden_size)
            # x shape: (num_tokens, hidden_size)
            if hasattr(self, "fc31_act_scale") and self.fc31_act_scale is not None:
                x = x * self.fc31_act_scale

            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            if pad_size > 0:
                x = torch.nn.functional.pad(x, (0, pad_size))

            x_row = x.shape[0]
            x, x_sf = self.op_backend.fp4_quantize(
                x, self.fc31_input_scale, self.scaling_vector_size, False, False
            )
        # All three branches produce scales today, but the W4A16 sibling
        # returns ``None`` here, so the absence is passed through.
        return x, None if x_sf is None else x_sf.view(x_row, -1)


class TRTLLMGenW4a16Mxfp4Base(TRTLLMGenFp4BlockScaleBase):
    """MXFP4 weights, bfloat16 activations."""

    supports_gptoss_style = True
    needs_zero_expert_bias = True

    def _get_quant_method(self):
        return W4A16MXFP4TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        # Weight-only: the activation is padded to the packed weight width and
        # stays bfloat16, so there is no scaling factor to hand back.
        pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad_size)), None


class TRTLLMGenW4a8Mxfp4Mxfp8Base(TRTLLMGenFp4BlockScaleBase):
    """MXFP4 weights, MXFP8 activations, group-32 block scales."""

    supports_gptoss_style = True
    # Group size of the ``Bmm_MxE4m3_MxE2m1MxE4m3_..._siTuGlu_*`` cubins; see
    # ``TRTLLMGenNvfp4Base`` on why ``supports_situ`` is not set alongside it.
    situ_scaling_vector_size = 32
    needs_zero_expert_bias = True

    def _situ_tp_weight_alignment(self) -> int:
        """Whole group-32 scale groups per rank; the loader pads to 128 itself."""
        return W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.scaling_vector_size

    def _get_quant_method(self):
        return W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        x, x_sf = self.op_backend.mxfp8_quantize(
            x, False, alignment=self.quant_method.input_hidden_alignment
        )
        return x, x_sf.view(x.shape[0], -1)
