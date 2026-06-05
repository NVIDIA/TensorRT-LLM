# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Marlin-based MoE backend for NVFP4 on SM90 (Hopper).

Uses a fused ``marlin_nvfp4_moe_gemm`` CUDA kernel that processes ALL experts
in a single launch.  W4A16 approach: BF16 activations + FP4 weights,
dequantized FP4→BF16 in registers, using BF16 m16n8k16 MMA.
No activation quantization overhead.  In-kernel topk_weights multiplication.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...utils import ActivationType, is_gated_activation, relu2
from .fused_moe_cutlass import CutlassFusedMoE
from .interface import _warn_and_return
from .quantization import NVFP4CutlassFusedMoEMethod, float4_sf_dtype

# Block size for moe_align_block_size — must match TILE_M in the kernel
_MOE_BLOCK_SIZE = 16


def _has_fused_moe_kernel() -> bool:
    """Check if the fused marlin_nvfp4_moe_gemm op is available."""
    return hasattr(torch.ops.trtllm, "marlin_nvfp4_moe_gemm")


class NVFP4MarlinFusedMoEMethod(NVFP4CutlassFusedMoEMethod):
    """NVFP4 MoE quantization method for the Marlin backend.

    Inherits weight loading from the CUTLASS method, then transforms the loaded
    weights to Marlin tiled format in ``post_load_weights``.

    The Marlin kernel is W4A16 (BF16 activations, no activation quantization),
    so global_scale must be the raw ``weight_scale_2`` — not the CUTLASS alpha
    which folds in ``input_scale``.  We intercept alpha loading to save the
    raw ``weight_scale_2`` values.
    """

    def load_expert_fc31_alpha_nvfp4(
        self, w1_weight_scale_2, w3_weight_scale_2, final_fc31_input_scale, dst_fc31_alpha
    ):
        # Store raw weight_scale_2 for Marlin (W4A16: no input_scale needed).
        w1_ws2 = w1_weight_scale_2[...].reshape([])
        dst_fc31_alpha.copy_(w1_ws2)

    def load_expert_fc2_alpha_nvfp4(self, w2_weight_scale_2, final_fc2_input_scale, dst_w2_alpha):
        w2_ws2 = w2_weight_scale_2[...].reshape([])
        dst_w2_alpha.copy_(w2_ws2)

    def post_load_weights(self, module):
        """Transform CUTLASS-format NVFP4 weights to Marlin tiled format."""
        from tensorrt_llm.quantization.utils import marlin_utils

        # Standard CUTLASS loading (swizzles scales, computes alpha, etc.)
        super().post_load_weights(module)

        num_experts = module.expert_size_per_partition
        hidden_size = module.hidden_size
        intermediate_size = module.intermediate_size_per_partition
        is_act_and_mul = module.intermediate_size_expand_ratio == 2
        group_size = module.scaling_vector_size  # 16

        # Actual (unpadded) dimensions
        N1 = intermediate_size * (2 if is_act_and_mul else 1)
        K1 = hidden_size
        N2 = hidden_size
        K2 = intermediate_size

        def unswizzle_scales(scale_3d, N_actual, K_actual):
            """Unswizzle packed int32 scales -> FP8 [num_experts, N, num_groups]."""
            num_groups = K_actual // group_size
            result = []
            for i in range(num_experts):
                # [N_padded, K//64] int32 -> float4_sf_dtype -> reverse -> FP8
                s_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                    scale_3d[i].view(float4_sf_dtype)
                )
                s_fp8 = s_unswizzled.view(torch.float8_e4m3fn)[:N_actual, :num_groups]
                result.append(s_fp8.unsqueeze(0))
            return torch.cat(result, 0)

        w13_scale = unswizzle_scales(module.w3_w1_weight_scale, N1, K1)
        w2_scale = unswizzle_scales(module.w2_weight_scale, N2, K2)
        w13_weight = module.w3_w1_weight.view(torch.uint8)[:, :N1, : K1 // 2].contiguous()
        w2_weight = module.w2_weight.view(torch.uint8)[:, :N2, : K2 // 2].contiguous()

        w13_gs = module.fc31_alpha.data.clone()
        w2_gs = module.fc2_alpha.data.clone()

        (w13, w13_s, w13_gs, w2, w2_s, w2_gs) = marlin_utils.prepare_nvfp4_moe_weights_for_marlin(
            w13=w13_weight,
            w13_scale=w13_scale,
            w13_global_scale=w13_gs,
            w2=w2_weight,
            w2_scale=w2_scale,
            w2_global_scale=w2_gs,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            num_experts=num_experts,
            is_act_and_mul=is_act_and_mul,
            param_dtype=torch.bfloat16,
        )

        for name in (
            "w3_w1_weight",
            "w2_weight",
            "w3_w1_weight_scale",
            "w2_weight_scale",
            "fc31_alpha",
            "fc2_alpha",
        ):
            getattr(module, name).data.untyped_storage().resize_(0)

        module.w3_w1_weight = nn.Parameter(w13.view(torch.int64), requires_grad=False)
        module.w3_w1_weight_scale = nn.Parameter(w13_s.view(torch.int32), requires_grad=False)
        module.fc31_alpha = nn.Parameter(w13_gs, requires_grad=False)
        module.w2_weight = nn.Parameter(w2.view(torch.int64), requires_grad=False)
        module.w2_weight_scale = nn.Parameter(w2_s.view(torch.int32), requires_grad=False)
        module.fc2_alpha = nn.Parameter(w2_gs, requires_grad=False)


class MarlinFusedMoE(CutlassFusedMoE):
    """MoE backend using Marlin W4A16 NVFP4 GEMM for SM90 (Hopper).

    Uses ``marlin_nvfp4_moe_gemm`` with BF16 activations to process all experts
    in a single kernel launch via sorted token dispatch. In-kernel topk_weights
    multiplication eliminates separate scatter-weight step. CUDA-graph
    compatible. Requires the fused kernel to be built (no fallback path).
    """

    _QUANT_SUPPORT_TABLE = {
        QuantAlgo.NVFP4: {
            "sm_constraint": ("in", {90}),
            "dtypes": {torch.bfloat16},
        },
    }

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        sm_version = get_sm_version()

        if quant_algo != QuantAlgo.NVFP4:
            return _warn_and_return(
                f"MarlinFusedMoE only supports NVFP4 (got quant_algo={quant_algo})"
            )

        if sm_version != 90:
            return _warn_and_return(
                f"MarlinFusedMoE only supports SM90 (Hopper), got SM{sm_version}"
            )

        if swiglu_gptoss_style:
            return _warn_and_return("MarlinFusedMoE does not support swiglu_gptoss_style")

        if dtype_activation != torch.bfloat16:
            return _warn_and_return(
                f"MarlinFusedMoE W4A16 requires bfloat16 activations, got {dtype_activation}"
            )

        return True, None

    def quantize_input(
        self, x: torch.Tensor | Fp4QuantizedTensor, post_quant_comm: bool = True, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        return x, None

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4():
            assert self.moe_backend == "MARLIN", (
                "MarlinFusedMoE only supports NVFP4, got {self.moe_backend}"
            )
            return NVFP4MarlinFusedMoEMethod()
        raise ValueError(f"MarlinFusedMoE only supports NVFP4, got {self.quant_config}")

    def _supports_load_balancer(self) -> bool:
        return False

    def _apply_activation(self, gemm1_out: torch.Tensor) -> torch.Tensor:
        """Apply the activation function to the gemm1 output.

        For gated activations (SwiGLU, GeGLU), gemm1_out has shape
        [tokens, 2 * intermediate_size] — split into gate + up.
        For non-gated activations (Relu2, Relu), gemm1_out has shape
        [tokens, intermediate_size] — apply element-wise.

        Returns [tokens, intermediate_size] in all cases.
        """
        inter_size = self.intermediate_size_per_partition
        if is_gated_activation(self.activation_type):
            gate = gemm1_out[:, :inter_size]
            up = gemm1_out[:, inter_size : 2 * inter_size]
            if self.activation_type == ActivationType.Geglu:
                return F.gelu(gate) * up
            else:
                return F.silu(gate) * up  # SwiGLU
        else:
            if self.activation_type == ActivationType.Relu2:
                return relu2(gemm1_out)
            else:
                return F.relu(gemm1_out)

    def _ensure_workspace(self, device: torch.device):
        """Lazily allocate workspace tensor for Marlin kernel."""
        if not hasattr(self, "_marlin_workspace") or self._marlin_workspace is None:
            props = torch.cuda.get_device_properties(device)
            sms = props.multi_processor_count
            max_blocks_per_sm = 4
            self._marlin_workspace = torch.zeros(
                sms * max_blocks_per_sm, dtype=torch.int32, device=device
            )
        return self._marlin_workspace

    # ====================================================================
    # Main entry point
    # ====================================================================

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        is_sf_swizzled: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: Optional[bool] = None,
    ) -> torch.Tensor:
        assert output_dtype is None or output_dtype == torch.bfloat16
        assert _has_fused_moe_kernel(), (
            "marlin_nvfp4_moe_gemm is not available. Rebuild TensorRT-LLM "
            "with the fused Marlin MoE kernel for NVFP4."
        )
        assert x.dtype == torch.bfloat16

        output_dtype = torch.bfloat16

        num_tokens = x.shape[0]
        top_k = token_selected_experts.shape[1]

        local_n = self.expert_size_per_partition
        if local_n != self.num_experts:
            slot_start = self.slot_start
            is_local = (token_selected_experts >= slot_start) & (
                token_selected_experts < slot_start + local_n
            )
            token_selected_experts = (token_selected_experts - slot_start).clamp(0, local_n - 1)
            if token_final_scales is None:
                token_final_scales = torch.ones(
                    num_tokens, top_k, dtype=torch.float32, device=x.device
                )
            token_final_scales = token_final_scales * is_local.to(token_final_scales.dtype)
        num_experts = local_n

        workspace = self._ensure_workspace(x.device)  # [num_sms * max_blocks_per_sm(4)] int32

        # Step 1: Sort tokens by expert assignment
        topk_ids = token_selected_experts.to(torch.int32).contiguous()
        max_num_tokens_padded = num_tokens * top_k + num_experts * _MOE_BLOCK_SIZE

        sorted_token_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=x.device)
        expert_ids_out = torch.empty(
            (max_num_tokens_padded + _MOE_BLOCK_SIZE - 1) // _MOE_BLOCK_SIZE,
            dtype=torch.int32,
            device=x.device,
        )
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=x.device)

        torch.ops.trtllm.moe_align_block_size(
            topk_ids,
            num_experts,
            _MOE_BLOCK_SIZE,
            sorted_token_ids,
            expert_ids_out,
            num_tokens_post_pad,
        )

        # Prepare topk_weights for in-kernel multiplication
        if token_final_scales is not None:
            topk_weights = token_final_scales.float().contiguous()
        else:
            topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=x.device)

        hidden_size = x.shape[1]
        k1 = hidden_size
        n1 = self.expand_intermediate_size_per_partition

        # Step 2: Fused gemm1 — ALL experts in ONE kernel launch (W4A16)
        gemm1_out = torch.ops.trtllm.marlin_nvfp4_moe_gemm(
            x.contiguous(),
            self.w3_w1_weight,
            b_scales=self.w3_w1_weight_scale,
            global_scale=self.fc31_alpha,
            workspace=workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids_out,
            num_tokens_past_padded=num_tokens_post_pad,
            topk_weights=topk_weights,
            moe_block_size=_MOE_BLOCK_SIZE,
            top_k=top_k,
            mul_topk_weights=False,  # Don't multiply weights in gemm1
            size_n=n1,
            size_k=k1,
            out_dtype=output_dtype,
            use_fp32_reduce=False,
        )  # [num_tokens * top_k, expand_intermediate]

        # Step 3: Activation (element-wise)
        hidden = self._apply_activation(gemm1_out)

        k2 = self.intermediate_size_per_partition
        n2 = self.unpadded_hidden_size

        # Step 4: Fused gemm2 — ALL experts in ONE kernel launch (W4A16)
        # hidden is [num_tokens * top_k, intermediate_size]. Each row is an
        # independent token-expert pair.  We re-sort with top_k=1 so the
        # kernel maps each row to the correct expert without expanding again.
        num_tokens_gemm2 = num_tokens * top_k

        # Build per-row expert assignment from the original topk_ids
        gemm2_topk_ids = topk_ids.reshape(-1, 1)[:num_tokens_gemm2].contiguous()

        max_padded_g2 = num_tokens_gemm2 + num_experts * _MOE_BLOCK_SIZE
        sorted_ids_g2 = torch.empty(max_padded_g2, dtype=torch.int32, device=x.device)
        expert_ids_g2 = torch.empty(
            (max_padded_g2 + _MOE_BLOCK_SIZE - 1) // _MOE_BLOCK_SIZE,
            dtype=torch.int32,
            device=x.device,
        )
        num_post_pad_g2 = torch.empty(1, dtype=torch.int32, device=x.device)

        torch.ops.trtllm.moe_align_block_size(
            gemm2_topk_ids,
            num_experts,
            _MOE_BLOCK_SIZE,
            sorted_ids_g2,
            expert_ids_g2,
            num_post_pad_g2,
        )

        # topk_weights for gemm2: flatten to [num_tokens*top_k, 1] for top_k=1
        topk_weights_g2 = topk_weights.reshape(-1, 1)[:num_tokens_gemm2].contiguous()

        gemm2_out = torch.ops.trtllm.marlin_nvfp4_moe_gemm(
            hidden.contiguous(),
            self.w2_weight,
            b_scales=self.w2_weight_scale,
            global_scale=self.fc2_alpha,
            workspace=workspace,
            sorted_token_ids=sorted_ids_g2,
            expert_ids=expert_ids_g2,
            num_tokens_past_padded=num_post_pad_g2,
            topk_weights=topk_weights_g2,
            moe_block_size=_MOE_BLOCK_SIZE,
            top_k=1,
            mul_topk_weights=True,
            size_n=n2,
            size_k=k2,
            out_dtype=output_dtype,
            use_fp32_reduce=False,
        )  # [num_tokens_gemm2, hidden_size]

        # Step 5: Scatter-reduce — sum weighted expert outputs.
        # gemm2_out rows correspond to flattened (token_idx * top_k + k) pairs.
        gemm2_out = gemm2_out[:num_tokens_gemm2, : self.unpadded_hidden_size]

        # Map each row back to its original token index
        row_indices = torch.arange(num_tokens_gemm2, device=x.device)
        orig_tokens = row_indices // top_k

        final_hidden_states = torch.zeros(
            (num_tokens, self.unpadded_hidden_size),
            dtype=output_dtype,
            device=x.device,
        )
        final_hidden_states.index_add_(0, orig_tokens, gemm2_out.to(output_dtype))

        return final_hidden_states
