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
MoE Op abstraction for supporting different MoE computation implementations.
This module provides a unified interface for different MoE ops (Cutlass, DeepGemm, etc.)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import get_sm_version

if TYPE_CHECKING:
    from ..interface import MoE


class MoEOp(ABC):
    """Abstract base class for MoE computation ops.

    This class provides a strategy pattern for different MoE computation implementations.
    It is used by MoE modules (like WideEPMoE) to delegate the actual computation.

    Note: MoEOp is NOT a MoE module itself, but a computation strategy.
    The actual MoE module (e.g., WideEPMoE) inherits from MoE and uses MoEOp
    for the computation implementation.
    """

    # Op-specific abstract methods
    @abstractmethod
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
        Finalize tactics for the MoE computation.
        For Cutlass op, this includes profiling and tactic selection.
        For DeepGemm op, this can be a no-op.

        Args:
            module: The MoE module containing MoE configurations
            tuner_input: Real input used for tuning (same shape/layout as non-alltoall)
            output_dtype: Output dtype for tuner run
            min_latency_mode: Whether to profile for min-latency path
            use_fused_finalize: Whether to use fused finalize
            tuner_top_k: Top-k value for tuning (Cutlass specific)
        """

    @abstractmethod
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
        Perform the actual MoE computation.

        Configuration parameters (tp_size, ep_size, swiglu params, etc.) are
        automatically extracted from the module parameter.

        Args:
            module: MoE module containing configuration and parameters.
            x: Input tensor
            token_selected_slots: Selected expert slots
            token_final_scales: Scaling factors
            w3_w1_weight: Fused gate and up projection weights
            w3_w1_bias: Optional bias
            w2_weight: Down projection weights
            w2_bias: Optional bias
            output_dtype: Output data type
            quant_scales: Quantization scales
            use_all_to_all: Whether to use all-to-all communication
            input_sf: Input scaling factor
            swizzled_input_sf: Whether input_sf is swizzled
            min_latency_mode: Use minimum latency optimizations
            use_fused_finalize: Use fused finalization
            tuner_num_tokens: Number of tokens for tuning
            tuner_top_k: Top-k value for tuning

        Returns:
            Computed MoE output tensor
        """

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
        Run the complete MoE computation pipeline.

        Configuration parameters are automatically extracted from the module.

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
            use_all_to_all: Whether to use all-to-all communication
            input_sf: Optional input scale factors for quantization
            swizzled_input_sf: Whether input scale factors are swizzled
            min_latency_mode: Use minimum latency optimizations
            use_fused_finalize: Use fused finalization
            tuner_num_tokens: Number of tokens for tuner input
            tuner_top_k: Top-k value for tuning

        Returns:
            Computed MoE output tensor
        """
        self.finalize_tactic(module, input, output_dtype, min_latency_mode,
                             use_fused_finalize, tuner_top_k)

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


class MoEOpSelector:
    """
    Utility class for selecting the appropriate MoE op based on
    hardware capabilities and quantization configuration.

    This class implements the strategy pattern for op selection,
    choosing between Cutlass and DeepGemm implementations based on:
    - Hardware capabilities (SM version)
    - Quantization configuration (block FP8 support)
    """

    @staticmethod
    def select_op(module: 'MoE') -> MoEOp:
        """
        Select the appropriate MoE op based on module configuration.

        Selection criteria:
        - Blackwell (SM100) with block FP8 quantization -> DeepGemm op
        - All other configurations -> Cutlass op

        Args:
            module: The MoE module containing configuration information

        Returns:
            MoEOp: Selected op instance (CutlassMoEOp or DeepGemmMoEOp)

        Example:
            >>> op = MoEOpSelector.select_op(moe_module)
            >>> output = op.run_moe(input, ...)
        """
        from .moe_op_cutlass import CutlassMoEOp
        from .moe_op_deepgemm import DeepGemmMoEOp

        # Check if we should use DeepGemm op
        # Blackwell has SM version 100
        is_blackwell = get_sm_version() == 100
        has_block_fp8 = module.has_deepseek_fp8_block_scales

        if is_blackwell and has_block_fp8:
            # Use DeepGemm op for Blackwell with block FP8
            return DeepGemmMoEOp()
        else:
            # Use Cutlass op for all other cases
            return CutlassMoEOp()
