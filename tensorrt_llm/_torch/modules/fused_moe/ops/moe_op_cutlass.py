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
Cutlass-based MoE op implementation.
"""

from typing import TYPE_CHECKING, List, Optional

import torch

from .moe_op import MoEOp

if TYPE_CHECKING:
    from ..interface import MoE


class CutlassMoEOp(MoEOp):
    """Cutlass-based MoE op using torch.ops.trtllm.fused_moe."""

    def __init__(self):
        """Initialize the Cutlass op."""
        super().__init__()
        self.moe_runner = None
        self.gemm_tactics = None

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
        Finalize tactics for Cutlass MoE by profiling and selecting optimal GEMM tactics.
        """

        # Import necessary modules for profiling
        from ....custom_ops.torch_custom_ops import AutoTuner, MoERunner

        # Use real tuner_input rather than dummy input
        assert tuner_input is not None, "tuner_input must be provided to finalize_tactic"
        if tuner_top_k is None:
            tuner_top_k = getattr(module.routing_method, 'experts_per_token', 1)

        # Determine view dtype for weights to match runtime quantization layout
        weight_view_dtype = module.w3_w1_weight.dtype
        if getattr(module, 'has_w4afp8', False):
            weight_view_dtype = torch.quint4x2
        elif module.has_w4a16_mxfp4:
            weight_view_dtype = torch.uint8

        # Create MoERunner for profiling
        if self.moe_runner is None:
            self.moe_runner = MoERunner(
                x_dtype=tuner_input.dtype,
                weight_dtype=weight_view_dtype,
                output_dtype=output_dtype,
                top_k=tuner_top_k,
                tp_size=module.tp_size,
                tp_rank=module.tp_rank,
                ep_size=module.ep_size,
                ep_rank=module.ep_rank,
                cluster_size=module.cluster_size,
                cluster_rank=module.cluster_rank,
                use_deepseek_fp8_block_scale=module.
                has_deepseek_fp8_block_scales,
                use_w4_group_scaling=getattr(module, 'has_w4afp8', False),
                use_int8_woq_per_channel=getattr(module,
                                                 'has_int8_woq_per_channel',
                                                 False),
                use_mxfp8_act_scaling=getattr(module, 'has_mxfp8_act_scaling',
                                              False),
                min_latency_mode=min_latency_mode,
                use_fused_finalize=use_fused_finalize,
            )

        # Set tuning configuration
        MoERunner.tuning_config.tune_max_num_tokens = getattr(
            module, 'tune_max_num_tokens', 8192)

        # Get AutoTuner for tactic selection
        tuner = AutoTuner.get()

        # Profile and select tactics (GEMM1)
        _, gemm_tactic_1 = tuner.choose_one(
            "trtllm::fused_moe::gemm1",
            [self.moe_runner],
            MoERunner.tuning_config,
            [
                tuner_input,
                module.w3_w1_weight.view(weight_view_dtype),
                getattr(module, 'w3_w1_bias', None),
                module.w2_weight.view(weight_view_dtype),
                getattr(module, 'w2_bias', None),
            ],
            gemm_idx=1,
        )

        # Profile and select tactics (GEMM2)
        _, gemm_tactic_2 = tuner.choose_one(
            "trtllm::fused_moe::gemm2",
            [self.moe_runner],
            MoERunner.tuning_config,
            [
                tuner_input,
                module.w3_w1_weight.view(weight_view_dtype),
                getattr(module, 'w3_w1_bias', None),
                module.w2_weight.view(weight_view_dtype),
                getattr(module, 'w2_bias', None),
            ],
            gemm_idx=2,
        )

        # Store selected tactics
        self.gemm_tactics = [gemm_tactic_1, gemm_tactic_2]

    def compute_moe(
            self,
            module: 'MoE',  # Now required as first parameter
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
        Compute MoE using Cutlass op with MoERunner.
        """
        # Extract parameters from module
        tp_size = module.tp_size
        tp_rank = module.tp_rank
        ep_size = module.ep_size
        ep_rank = module.ep_rank
        cluster_size = module.cluster_size
        cluster_rank = module.cluster_rank
        use_all_to_all = use_all_to_all
        swiglu_alpha = module.swiglu_alpha
        swiglu_beta = module.swiglu_beta
        swiglu_limit = module.swiglu_limit
        use_w4_group_scaling = getattr(module, 'has_w4afp8', False)

        # Determine weight dtype for view operation if needed
        weight_dtype = w3_w1_weight.dtype
        if use_w4_group_scaling and weight_dtype != torch.quint4x2:
            weight_dtype = torch.quint4x2

        # Validate that tactics have been finalized
        if self.gemm_tactics is None or len(self.gemm_tactics) == 0:
            raise RuntimeError(
                "GEMM tactics have not been finalized. "
                "Call finalize_tactic() before compute_moe() or use run_moe() instead."
            )

        if self.moe_runner is None:
            raise RuntimeError(
                "MoERunner has not been initialized. "
                "Call finalize_tactic() before compute_moe() or use run_moe() instead."
            )

        # Select the appropriate run method based on latency mode
        run_moe = self.moe_runner.fused_moe_runner.run_moe_min_latency if min_latency_mode else self.moe_runner.fused_moe_runner.run_moe

        # Get unpadded_hidden_size from module if available, otherwise use hidden_size
        # For now it is the user's responsibility to set unpadded_hidden_size.
        # DeepGemmFusedMoE and WideEPMoE both have unpadded_hidden_size.
        unpadded_hidden_size = getattr(module, 'unpadded_hidden_size',
                                       x.shape[1])

        # Run the actual MoE computation
        output = run_moe(
            x,
            token_selected_slots,
            token_final_scales,
            w3_w1_weight.view(weight_dtype),
            w3_w1_bias,
            w2_weight.view(weight_dtype),
            w2_bias,
            quant_scales,
            input_sf,
            swizzled_input_sf,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
            use_all_to_all,
            min_latency_mode,
            self.gemm_tactics,
            unpadded_hidden_size,
        )

        # Return output based on latency mode
        return output if min_latency_mode else [output]

    def run_moe(
            self,
            module: 'MoE',
            # Input tensors
            input: torch.Tensor,
            token_selected_slots: torch.Tensor,
            token_final_scales: torch.Tensor,
            w3_w1_weight: torch.Tensor,
            w3_w1_bias: Optional[torch.Tensor],
            w2_weight: torch.Tensor,
            w2_bias: Optional[torch.Tensor],
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
        Run the complete MoE computation pipeline for Cutlass op.

        This override handles the specific tuner_input logic needed for Cutlass.

        Args:
            module: MoE module containing configuration
            input: Input tensor to the MoE layer
            token_selected_slots: Selected expert slots for each token
            token_final_scales: Final scaling factors for each token
            w3_w1_weight: Concatenated weights for w3 and w1 projections
            w3_w1_bias: Optional bias for w3/w1 projections
            w2_weight: Weight for w2 projection
            w2_bias: Optional bias for w2 projection
            output_dtype: Desired output data type
            quant_scales: Quantization scales for weights
            input_sf: Optional input scale factors for quantization
            swizzled_input_sf: Whether input scale factors are swizzled
            min_latency_mode: Use minimum latency optimizations
            use_fused_finalize: Use fused finalization
            tuner_num_tokens: Number of tokens for tuner input
            tuner_top_k: Top-k value for tuning
            use_all_to_all: Whether to use all-to-all communication

        Returns:
            Computed MoE output tensor
        """
        use_all_to_all = use_all_to_all

        # Compute tuner_input per fused_moe logic
        if use_all_to_all:
            assert tuner_num_tokens is not None
            assert tuner_top_k is not None
            tuner_input = input[:tuner_num_tokens]
        else:
            assert tuner_num_tokens is None
            assert tuner_top_k is None
            tuner_input = input
            tuner_top_k = token_selected_slots.size(1)

        self.finalize_tactic(module, tuner_input, output_dtype,
                             min_latency_mode, use_fused_finalize, tuner_top_k)

        # Call compute_moe with module
        return self.compute_moe(module=module,
                                x=input,
                                token_selected_slots=token_selected_slots,
                                token_final_scales=token_final_scales,
                                w3_w1_weight=w3_w1_weight,
                                w3_w1_bias=w3_w1_bias,
                                w2_weight=w2_weight,
                                w2_bias=w2_bias,
                                output_dtype=output_dtype,
                                quant_scales=quant_scales,
                                use_all_to_all=use_all_to_all,
                                input_sf=input_sf,
                                swizzled_input_sf=swizzled_input_sf,
                                min_latency_mode=min_latency_mode,
                                use_fused_finalize=use_fused_finalize,
                                tuner_num_tokens=tuner_num_tokens,
                                tuner_top_k=tuner_top_k,
                                **kwargs)
