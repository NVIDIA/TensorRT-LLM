"""
MoE Backend abstraction for supporting different MoE computation implementations.
This module provides a unified interface for different MoE backends (Cutlass, DeepGemm, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from tensorrt_llm._utils import get_sm_version


class MoEBackend(ABC):
    """Abstract base class for MoE computation backends.

    This class provides a strategy pattern for different MoE computation implementations.
    It is used by MoE modules (like WideEPMoE) to delegate the actual computation.

    Note: MoEBackend is NOT a MoE module itself, but a computation strategy.
    The actual MoE module (e.g., WideEPMoE) inherits from MoE and uses MoEBackend
    for the computation implementation.
    """

    # Backend-specific abstract methods
    @abstractmethod
    def finalize_tactic(
        self,
        module: Any,
        tuner_input: torch.Tensor,
        output_dtype: torch.dtype,
        min_latency_mode: bool = False,
        tuner_top_k: Optional[int] = None,
    ) -> None:
        """
        Finalize tactics for the MoE computation.
        For Cutlass backend, this includes profiling and tactic selection.
        For DeepGemm backend, this can be a no-op.

        Args:
            module: The MoE module containing weights and configurations
            tuner_input: Real input used for tuning (same shape/layout as non-alltoall)
            output_dtype: Output dtype for tuner run
            min_latency_mode: Whether to profile for min-latency path
            tuner_top_k: Top-k value for tuning (Cutlass specific)
        """

    @abstractmethod
    def compute_moe(
            self,
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
            input_sf: Optional[torch.Tensor] = None,
            swizzled_input_sf: bool = True,
            # SwiGLU parameters (optional)
            swiglu_alpha: Optional[torch.Tensor] = None,
            swiglu_beta: Optional[torch.Tensor] = None,
            swiglu_limit: Optional[torch.Tensor] = None,
            # Parallel configuration
            tp_size: int = 1,
            tp_rank: int = 0,
            ep_size: int = 1,
            ep_rank: int = 0,
            cluster_size: int = 1,
            cluster_rank: int = 0,
            enable_alltoall: bool = False,
            # Quantization flags
            use_deepseek_fp8_block_scale: bool = False,
            use_w4_group_scaling: bool = False,
            use_int8_woq_per_channel: bool = False,
            use_mxfp8_act_scaling: bool = False,
            # Performance tuning
            min_latency_mode: bool = False,
            use_fused_finalize: bool = True,
            tune_max_num_tokens: int = 8192,
            tuner_num_tokens: Optional[int] = None,
            tuner_top_k: Optional[int] = None,
            module: Optional[Any] = None,
            **kwargs) -> torch.Tensor:
        """
        Perform the actual MoE computation with full parameter support.

        This method should be compatible with the original torch.ops.trtllm.fused_moe
        interface to ensure backward compatibility and full feature support.

        Returns:
            Computed MoE output tensor
        """

    @abstractmethod
    def run_moe(
            self,
            # Positional arguments (same order as torch.ops.trtllm.fused_moe)
            input: torch.Tensor,
            token_selected_slots: torch.Tensor,
            token_final_scales: torch.Tensor,
            w3_w1_weight: torch.Tensor,
            w3_w1_bias: Optional[torch.Tensor],
            w2_weight: torch.Tensor,
            w2_bias: Optional[torch.Tensor],
            output_dtype: torch.dtype,
            # Keyword arguments
            quant_scales: List[torch.Tensor],
            input_sf: Optional[torch.Tensor] = None,
            swizzled_input_sf: bool = True,
            swiglu_alpha: Optional[torch.Tensor] = None,
            swiglu_beta: Optional[torch.Tensor] = None,
            swiglu_limit: Optional[torch.Tensor] = None,
            tp_size: int = 1,
            tp_rank: int = 0,
            ep_size: int = 1,
            ep_rank: int = 0,
            cluster_size: int = 1,
            cluster_rank: int = 0,
            enable_alltoall: bool = False,
            use_deepseek_fp8_block_scale: bool = False,
            use_w4_group_scaling: bool = False,
            use_int8_woq_per_channel: bool = False,
            use_mxfp8_act_scaling: bool = False,
            min_latency_mode: bool = False,
            use_fused_finalize: bool = True,
            tune_max_num_tokens: int = 8192,
            tuner_num_tokens: Optional[int] = None,
            tuner_top_k: Optional[int] = None,
            module: Optional[
                Any] = None,  # Module reference for accessing properties
            **kwargs) -> torch.Tensor:
        """
        Run the complete MoE computation pipeline.

        This method provides a unified interface compatible with torch.ops.trtllm.fused_moe,
        handling both tactic finalization and the actual MoE computation.

        Args:
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
            swiglu_alpha/beta/limit: Optional SwiGLU activation parameters
            tp_size/tp_rank: Tensor parallel configuration
            ep_size/ep_rank: Expert parallel configuration
            cluster_size/cluster_rank: Cluster configuration
            enable_alltoall: Whether to use alltoall communication
            use_deepseek_fp8_block_scale: Enable DeepSeek FP8 block scaling
            use_w4_group_scaling: Enable W4 group scaling
            use_int8_woq_per_channel: Enable INT8 weight-only quantization
            use_mxfp8_act_scaling: Enable MXFP8 activation scaling
            min_latency_mode: Use minimum latency optimizations
            use_fused_finalize: Use fused finalization
            tune_max_num_tokens: Maximum tokens for tuning
            tuner_num_tokens: Number of tokens for tuner input (alltoall mode)
            tuner_top_k: Top-k value for tuning (alltoall mode)
            module: Optional MoE module reference for accessing properties

        Returns:
            Computed MoE output tensor
        """

        self.finalize_tactic(module, input, output_dtype, min_latency_mode,
                             tuner_top_k)

        # Call compute_moe with all parameters
        return self.compute_moe(
            x=input,
            token_selected_slots=token_selected_slots,
            token_final_scales=token_final_scales,
            w3_w1_weight=w3_w1_weight,
            w3_w1_bias=w3_w1_bias,
            w2_weight=w2_weight,
            w2_bias=w2_bias,
            output_dtype=output_dtype,
            quant_scales=quant_scales,
            input_sf=input_sf,
            swizzled_input_sf=swizzled_input_sf,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_int8_woq_per_channel=use_int8_woq_per_channel,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            use_fused_finalize=use_fused_finalize,
            tune_max_num_tokens=tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            module=module,
            **kwargs)


class MoeCutlassBackend(MoEBackend):
    """Cutlass-based MoE backend using torch.ops.trtllm.fused_moe."""

    def __init__(self):
        """Initialize the Cutlass backend."""
        super().__init__()
        self.moe_runner = None
        self.gemm_tactics = None

    def finalize_tactic(
        self,
        module: Any,
        tuner_input: torch.Tensor,
        output_dtype: torch.dtype,
        min_latency_mode: bool = False,
        tuner_top_k: Optional[int] = None,
    ) -> None:
        """
        Finalize tactics for Cutlass MoE by profiling and selecting optimal GEMM tactics.
        """

        # Import necessary modules for profiling
        from ...custom_ops.torch_custom_ops import AutoTuner, MoERunner

        # Use real tuner_input rather than dummy input
        assert tuner_input is not None, "tuner_input must be provided to finalize_tactic"
        if tuner_top_k is None:
            tuner_top_k = getattr(module.routing_method, 'experts_per_token', 1)

        # Determine view dtype for weights to match runtime quantization layout
        weight_view_dtype = module.w3_w1_weight.dtype
        if getattr(module, 'has_w4afp8', False):
            weight_view_dtype = torch.quint4x2
        elif getattr(module, 'has_w4a16_mxfp4', False):
            weight_view_dtype = torch.uint8

        # Create MoERunner for profiling
        if self.moe_runner is None:
            self.moe_runner = MoERunner(
                x_dtype=tuner_input.dtype,
                weight_dtype=module.w3_w1_weight.dtype,
                output_dtype=output_dtype,
                top_k=tuner_top_k,
                tp_size=module.tp_size,
                tp_rank=module.tp_rank,
                ep_size=module.ep_size,
                ep_rank=module.ep_rank,
                cluster_size=module.cluster_size,
                cluster_rank=module.cluster_rank,
                use_deepseek_fp8_block_scale=getattr(
                    module, 'has_deepseek_fp8_block_scales', False),
                use_w4_group_scaling=getattr(module, 'has_w4afp8', False),
                use_int8_woq_per_channel=getattr(module,
                                                 'has_int8_woq_per_channel',
                                                 False),
                use_mxfp8_act_scaling=getattr(module, 'has_mxfp8_act_scaling',
                                              False),
                min_latency_mode=min_latency_mode,
                use_fused_finalize=getattr(module, 'use_fused_finalize', False),
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
            input_sf: Optional[torch.Tensor] = None,
            swizzled_input_sf: bool = True,
            # SwiGLU parameters (optional)
            swiglu_alpha: Optional[torch.Tensor] = None,
            swiglu_beta: Optional[torch.Tensor] = None,
            swiglu_limit: Optional[torch.Tensor] = None,
            # Parallel configuration
            tp_size: int = 1,
            tp_rank: int = 0,
            ep_size: int = 1,
            ep_rank: int = 0,
            cluster_size: int = 1,
            cluster_rank: int = 0,
            enable_alltoall: bool = False,
            # Quantization flags
            use_deepseek_fp8_block_scale: bool = False,
            use_w4_group_scaling: bool = False,
            use_int8_woq_per_channel: bool = False,
            use_mxfp8_act_scaling: bool = False,
            # Performance tuning
            min_latency_mode: bool = False,
            use_fused_finalize: bool = True,
            tune_max_num_tokens: int = 8192,
            tuner_num_tokens: Optional[int] = None,
            tuner_top_k: Optional[int] = None,
            module: Optional[Any] = None,
            **kwargs) -> torch.Tensor:
        """
        Compute MoE using Cutlass backend with MoERunner.
        """
        # Import necessary modules

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
            enable_alltoall,
            min_latency_mode,
            self.gemm_tactics,
        )

        # Return output based on latency mode
        return output if min_latency_mode else [output]

    def run_moe(
            self,
            input: torch.Tensor,
            token_selected_slots: torch.Tensor,
            token_final_scales: torch.Tensor,
            w3_w1_weight: torch.Tensor,
            w3_w1_bias: Optional[torch.Tensor],
            w2_weight: torch.Tensor,
            w2_bias: Optional[torch.Tensor],
            output_dtype: torch.dtype,
            # Keyword arguments
            quant_scales: List[torch.Tensor],
            input_sf: Optional[torch.Tensor] = None,
            swizzled_input_sf: bool = True,
            swiglu_alpha: Optional[torch.Tensor] = None,
            swiglu_beta: Optional[torch.Tensor] = None,
            swiglu_limit: Optional[torch.Tensor] = None,
            tp_size: int = 1,
            tp_rank: int = 0,
            ep_size: int = 1,
            ep_rank: int = 0,
            cluster_size: int = 1,
            cluster_rank: int = 0,
            enable_alltoall: bool = False,
            use_deepseek_fp8_block_scale: bool = False,
            use_w4_group_scaling: bool = False,
            use_int8_woq_per_channel: bool = False,
            use_mxfp8_act_scaling: bool = False,
            min_latency_mode: bool = False,
            use_fused_finalize: bool = True,
            tune_max_num_tokens: int = 8192,
            tuner_num_tokens: Optional[int] = None,
            tuner_top_k: Optional[int] = None,
            module: Optional[
                Any] = None,  # Module reference for accessing properties
            **kwargs) -> torch.Tensor:
        """
        Run the complete MoE computation pipeline.

        This method provides a unified interface compatible with torch.ops.trtllm.fused_moe,
        handling both tactic finalization and the actual MoE computation.

        Args:
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
            swiglu_alpha/beta/limit: Optional SwiGLU activation parameters
            tp_size/tp_rank: Tensor parallel configuration
            ep_size/ep_rank: Expert parallel configuration
            cluster_size/cluster_rank: Cluster configuration
            enable_alltoall: Whether to use alltoall communication
            use_deepseek_fp8_block_scale: Enable DeepSeek FP8 block scaling
            use_w4_group_scaling: Enable W4 group scaling
            use_int8_woq_per_channel: Enable INT8 weight-only quantization
            use_mxfp8_act_scaling: Enable MXFP8 activation scaling
            min_latency_mode: Use minimum latency optimizations
            use_fused_finalize: Use fused finalization
            tune_max_num_tokens: Maximum tokens for tuning
            tuner_num_tokens: Number of tokens for tuner input (alltoall mode)
            tuner_top_k: Top-k value for tuning (alltoall mode)
            module: Optional MoE module reference for accessing properties

        Returns:
            Computed MoE output tensor
        """
        # Compute tuner_input per fused_moe logic
        if enable_alltoall:
            assert tuner_num_tokens is not None
            assert tuner_top_k is not None
            tuner_input = input[:tuner_num_tokens]
        else:
            assert tuner_num_tokens is None
            assert tuner_top_k is None
            tuner_input = input
            tuner_top_k = token_selected_slots.size(1)

        self.finalize_tactic(module, tuner_input, output_dtype,
                             min_latency_mode, tuner_top_k)

        # Call compute_moe with all parameters
        return self.compute_moe(
            x=input,
            token_selected_slots=token_selected_slots,
            token_final_scales=token_final_scales,
            w3_w1_weight=w3_w1_weight,
            w3_w1_bias=w3_w1_bias,
            w2_weight=w2_weight,
            w2_bias=w2_bias,
            output_dtype=output_dtype,
            quant_scales=quant_scales,
            input_sf=input_sf,
            swizzled_input_sf=swizzled_input_sf,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_int8_woq_per_channel=use_int8_woq_per_channel,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            use_fused_finalize=use_fused_finalize,
            tune_max_num_tokens=tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            module=module,
            **kwargs)


class MoeDeepGemmBackend(MoEBackend):
    """DeepGemm-based MoE backend for GB200 block FP8."""

    def __init__(self):
        """Initialize DeepGemm backend."""
        super().__init__()
        # Import DeepGemm specific functions
        import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
        from tensorrt_llm import deep_gemm
        self.deep_gemm = deep_gemm
        self.fp8_utils = fp8_utils

    def finalize_tactic(
        self,
        module: Any,
        tuner_input: torch.Tensor,
        output_dtype: torch.dtype,
        min_latency_mode: bool = False,
        tuner_top_k: Optional[int] = None,
    ) -> None:
        """
        No-op for DeepGemm backend as it doesn't require tactic profiling.
        DeepGemm uses static tactics and doesn't need runtime profiling.

        Args:
            module: The MoE module (unused for DeepGemm)
            tuner_input: Input tensor for tuning (unused for DeepGemm)
            output_dtype: Output dtype (unused for DeepGemm)
            min_latency_mode: Whether to use min-latency mode (unused for DeepGemm)
            tuner_top_k: Top-k value for tuning (unused for DeepGemm)
        """

    def _get_deepgemm_workspace(self, module: Any, m_max: int,
                                group_size: int) -> Dict[str, torch.Tensor]:
        """
        Get workspace for DeepGemm backend operations.

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
            input_sf: Optional[torch.Tensor] = None,
            swizzled_input_sf: bool = True,
            # SwiGLU parameters (optional)
            swiglu_alpha: Optional[torch.Tensor] = None,
            swiglu_beta: Optional[torch.Tensor] = None,
            swiglu_limit: Optional[torch.Tensor] = None,
            # Parallel configuration
            tp_size: int = 1,
            tp_rank: int = 0,
            ep_size: int = 1,
            ep_rank: int = 0,
            cluster_size: int = 1,
            cluster_rank: int = 0,
            enable_alltoall: bool = False,
            # Quantization flags
            use_deepseek_fp8_block_scale: bool = False,
            use_w4_group_scaling: bool = False,
            use_int8_woq_per_channel: bool = False,
            use_mxfp8_act_scaling: bool = False,
            # Performance tuning
            min_latency_mode: bool = False,
            use_fused_finalize: bool = True,
            tune_max_num_tokens: int = 8192,
            tuner_num_tokens: Optional[int] = None,
            tuner_top_k: Optional[int] = None,
            module: Optional[Any] = None,
            **kwargs) -> torch.Tensor:
        """
        Compute MoE using DeepGemm backend with block FP8 quantization.

        Note: This assumes the data has already been gathered/alltoall'd
        by the WideEP forward_chunk method.
        """
        # Import necessary functions for DeepGemm
        from .fused_moe_deepgemm import (masked_index_copy_group_quant_fp8,
                                         preprocess_after_permute, set_strides,
                                         triton_masked_index_gather)

        # Module is required for DeepGemm backend
        if module is None:
            raise ValueError(
                "Module reference is required for DeepGemm backend")

        # Not supported: min_latency_mode. Raise error if enabled.
        if min_latency_mode:
            raise NotImplementedError(
                "DeepGemm backend does not support min_latency_mode=True")

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

        self.deep_gemm.deepgemm_fp8_group_blockwise_gemm(
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

        self.deep_gemm.deepgemm_fp8_group_blockwise_gemm(
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
        final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
            permuted_data_tensor,
            None,  # biases (w2_bias could be added here if needed)
            token_final_scales,
            unpermuted_row_to_permuted_row_tensor,
            permuted_row_to_unpermuted_row_tensor,
            token_selected_slots,
            expert_first_token_offset_tensor,
            enable_alltoall,
            x.shape[0],  # num_rows
            x.shape[1],  # hidden_size
            module.routing_method.top_k if module else 1,
            expert_size_per_partition,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
        )

        return final_hidden_states


class MoEBackendSelection:
    """
    Utility class for selecting the appropriate MoE backend based on
    hardware capabilities and quantization configuration.

    This class implements the strategy pattern for backend selection,
    choosing between Cutlass and DeepGemm implementations based on:
    - Hardware capabilities (SM version)
    - Quantization configuration (block FP8 support)
    """

    @staticmethod
    def select_backend(module: Any) -> MoEBackend:
        """
        Select the appropriate MoE backend based on module configuration.

        Selection criteria:
        - Blackwell (SM100) with block FP8 quantization -> DeepGemm backend
        - All other configurations -> Cutlass backend

        Args:
            module: The MoE module containing configuration information
                   Expected attributes:
                   - has_deepseek_fp8_block_scales: Whether block FP8 is enabled

        Returns:
            MoEBackend: Selected backend instance (MoeCutlassBackend or MoeDeepGemmBackend)

        Example:
            >>> backend = MoEBackendSelection.select_backend(moe_module)
            >>> output = backend.run_moe(input, ...)
        """
        # Check if we should use DeepGemm backend
        # Blackwell has SM version 100
        is_blackwell = get_sm_version() == 100
        has_block_fp8 = (hasattr(module, 'has_deepseek_fp8_block_scales')
                         and module.has_deepseek_fp8_block_scales)

        if is_blackwell and has_block_fp8:
            # Use DeepGemm backend for Blackwell with block FP8
            return MoeDeepGemmBackend()
        else:
            # Use Cutlass backend for all other cases
            return MoeCutlassBackend()
