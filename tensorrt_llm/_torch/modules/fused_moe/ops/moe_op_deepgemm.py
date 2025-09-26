# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeepGemm-based MoE op implementation for GB200 block FP8.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from .moe_op import MoEOp

if TYPE_CHECKING:
    from ..interface import MoE


class DeepGemmMoEOp(MoEOp):
    """DeepGemm-based MoE op for GB200 block FP8."""

    def __init__(self):
        """Initialize DeepGemm op."""
        super().__init__()
        import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
        self.fp8_utils = fp8_utils

        from ..fused_moe_deepgemm import deepgemm_fp8_group_blockwise_gemm
        self.deepgemm_fp8_group_blockwise_gemm = deepgemm_fp8_group_blockwise_gemm

    def finalize_tactic(
        self,
        module: 'MoE',
        tuner_input: torch.Tensor,
        output_dtype: torch.dtype,
        min_latency_mode: bool = False,
        use_fused_finalize: bool = True,
        tuner_top_k: Optional[int] = None,
    ) -> None:
        """
        No-op for DeepGemm op as it doesn't require tactic profiling.

        Args:
            module: The MoE module
            tuner_input: Input tensor for tuning
            output_dtype: Output dtype
            min_latency_mode: Whether to use min-latency mode
            use_fused_finalize: Whether to use fused finalize
            tuner_top_k: Top-k value for tuning
        """

    def _get_deepgemm_workspace(self, module: 'MoE', m_max: int,
                                group_size: int) -> Dict[str, torch.Tensor]:
        """
        Get workspace for DeepGemm op operations.

        Args:
            module: The MoE module containing configuration
            m_max: Maximum number of tokens (aligned)
            group_size: Group size for quantization

        Returns:
            Dictionary containing workspace tensors
        """
        import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils

        # Get dimensions from module
        hidden_size = module.hidden_size
        intermediate_size = module.intermediate_size
        expert_size_per_partition = module.expert_size_per_partition

        # Calculate aligned dimensions
        m_padded = fp8_utils.align(m_max, 4)
        fp8_dim = max(hidden_size, intermediate_size)
        scale_k = fp8_utils.ceil_div(fp8_dim, group_size)
        scale_k_padded = fp8_utils.align(scale_k, 4)

        # Allocate workspace tensors
        workspace = {}

        # Workspace for FP8 activations
        workspace["workspace_0"] = torch.empty(
            (expert_size_per_partition * m_max * fp8_dim),
            dtype=torch.float8_e4m3fn,
            device='cuda')

        # Workspace for intermediate results
        workspace["workspace_1"] = torch.empty(
            (expert_size_per_partition * m_max *
             max(intermediate_size * 2, hidden_size)),
            dtype=torch.bfloat16,
            device='cuda')

        # Workspace for scaling factors
        workspace["workspace_sf"] = torch.empty(
            expert_size_per_partition * (scale_k_padded // 4) * m_padded,
            dtype=torch.int32,
            device='cuda')

        return workspace

    def compute_moe(
            self,
            module: 'MoE',
            # Input tensors
            x: torch.Tensor,
            token_selected_slots: torch.Tensor,
            token_final_scales: Optional[torch.Tensor],
            # Weight tensors
            w3_w1_weight: torch.Tensor,
            w3_w1_bias: Optional[torch.Tensor],
            w2_weight: torch.Tensor,
            w2_bias: Optional[torch.Tensor],
            # Output configuration
            output_dtype: torch.dtype,
            # Quantization parameters
            quant_scales: List[torch.Tensor],
            use_all_to_all: bool,
            input_sf: Optional[torch.Tensor] = None,
            swizzled_input_sf: bool = True,
            # Performance tuning (only runtime-variable parameters)
            min_latency_mode: bool = False,
            use_fused_finalize: bool = True,
            tuner_num_tokens: Optional[int] = None,
            tuner_top_k: Optional[int] = None,
            **kwargs) -> torch.Tensor:
        """
        Compute MoE using DeepGemm op with block FP8 quantization.

        Note: This assumes the data has already been gathered/alltoall'd
        by the WideEP forward_chunk method.
        """

        # Import necessary functions for DeepGemm
        from ..fused_moe_deepgemm import (masked_index_copy_group_quant_fp8,
                                          preprocess_after_permute, set_strides,
                                          triton_masked_index_gather)

        # Extract parameters from module
        tp_size = module.tp_size
        tp_rank = module.tp_rank
        ep_size = module.ep_size
        ep_rank = module.ep_rank
        cluster_size = module.cluster_size
        cluster_rank = module.cluster_rank
        use_all_to_all = use_all_to_all

        # Not supported: min_latency_mode. Raise error if enabled.
        if min_latency_mode:
            raise NotImplementedError(
                "DeepGemm op does not support min_latency_mode=True")

        # Get expert configuration from module
        expert_size_per_partition = module.expert_size_per_partition
        intermediate_size = module.intermediate_size
        hidden_size = x.shape[1]

        # Permute the data for expert-parallel processing
        (
            permuted_row_to_unpermuted_row_tensor,
            permuted_token_selected_experts_tensor,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            permuted_token_final_scales_tensor,
            unpermuted_row_to_permuted_row_tensor,
        ) = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_slots,
            token_final_scales,
            None,  # w3_w1_weight
            None,  # w2_weight
            None,  # quant_scales
            input_sf=input_sf,
            num_experts_on_rank=expert_size_per_partition,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            min_latency_mode=min_latency_mode,
            use_fp8_block_scaling=True,  # Always use block scaling for DeepGemm
        )

        if permuted_data_tensor.numel() == 0:
            return torch.zeros_like(x)

        # Preprocess for masked operations
        masked_m, token_to_expert_map = preprocess_after_permute(
            expert_first_token_offset_tensor, permuted_data_tensor)

        expected_m = (token_selected_slots.numel() + expert_size_per_partition -
                      1) // expert_size_per_partition

        # Get workspace for DeepGemm operations
        m_max = self.fp8_utils.align(x.shape[0], 128)
        workspace = self._get_deepgemm_workspace(module, m_max, 128)

        # Padding and quantization for first GEMM input
        m_padded = self.fp8_utils.align(m_max, 4)
        scale_k = self.fp8_utils.ceil_div(hidden_size, 128)
        scale_k_padded = self.fp8_utils.align(scale_k, 4)

        act_input_fp8 = set_strides(workspace["workspace_0"],
                                    expert_size_per_partition, m_max,
                                    hidden_size)
        act_input_sf = set_strides(workspace["workspace_sf"],
                                   expert_size_per_partition,
                                   scale_k_padded // 4, m_padded)

        # Quantize and copy input with masking
        act_input_sf = masked_index_copy_group_quant_fp8(
            act_input_fp8,
            act_input_sf,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            token_to_expert_map,
            group_size=128)

        # First grouped GEMM (w3 and w1)
        h1 = set_strides(workspace["workspace_1"], expert_size_per_partition,
                         m_max, intermediate_size * 2)

        self.deepgemm_fp8_group_blockwise_gemm(
            d=h1,
            a=act_input_fp8,
            b=w3_w1_weight,
            sfa=act_input_sf,
            sfb=quant_scales[0] if quant_scales else None,
            masked_m=masked_m,
            expected_m=expected_m,
        )

        # SiLU activation and quantization for second GEMM
        act_input_fp8 = set_strides(workspace["workspace_0"],
                                    expert_size_per_partition, m_max,
                                    intermediate_size)

        scale_k = self.fp8_utils.ceil_div(intermediate_size, 128)
        scale_k_padded = self.fp8_utils.align(scale_k, 4)
        act_input_sf = set_strides(workspace["workspace_sf"],
                                   expert_size_per_partition,
                                   scale_k_padded // 4, m_padded)

        act_input_sf = self.fp8_utils.silu_and_mul_masked_post_quant_fwd(
            output=act_input_fp8,
            output_scale=act_input_sf,
            input=h1,
            quant_group_size=128,
            masked_m=masked_m,
            scale_ue8m0=True)

        # Second grouped GEMM (w2)
        h3 = set_strides(workspace["workspace_1"], expert_size_per_partition,
                         m_max, hidden_size)

        self.deepgemm_fp8_group_blockwise_gemm(
            d=h3,
            a=act_input_fp8,
            b=w2_weight,
            sfa=act_input_sf,
            sfb=quant_scales[1] if quant_scales else None,
            masked_m=masked_m,
            expected_m=expected_m,
        )

        # Gather results back to original token order
        triton_masked_index_gather(permuted_data_tensor, h3,
                                   expert_first_token_offset_tensor,
                                   token_to_expert_map)

        # Finalize and scale the output
        # Get unpadded_hidden_size from module if available, otherwise use hidden_size
        # For now it is the user's responsibility to set unpadded_hidden_size.
        # DeepGemmFusedMoE and WideEPMoE both have unpadded_hidden_size.
        unpadded_hidden_size = getattr(module, 'unpadded_hidden_size',
                                       x.shape[1])

        final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
            permuted_data_tensor,
            None,  # biases (w2_bias could be added here if needed)
            token_final_scales,
            unpermuted_row_to_permuted_row_tensor,
            permuted_row_to_unpermuted_row_tensor,
            token_selected_slots,
            expert_first_token_offset_tensor,
            use_all_to_all,
            x.shape[0],  # num_rows
            x.shape[1],  # hidden_size
            unpadded_hidden_size,  # unpadded_hidden_size (may be different from hidden_size if padding was applied)
            module.routing_method.top_k if module else 1,  # experts_per_token
            expert_size_per_partition,  # num_experts_per_node
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
        )

        return final_hidden_states if min_latency_mode else [
            final_hidden_states
        ]
