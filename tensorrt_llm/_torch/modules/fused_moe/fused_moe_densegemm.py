# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm.quantization.utils import fp4_utils

from ...distributed import allgather
from ...memory_buffer_utils import get_memory_buffers
from ...model_config import ModelConfig
from ...utils import AuxStreamType, EventType, Fp4QuantizedTensor, swizzle_sf, unswizzle_sf
from .fused_moe_cutlass import CutlassFusedMoE
from .interface import AlltoallMethodType
from .quantization import MoEWeightLoadingMode, NVFP4CuteDslFusedMoEMethod
from .routing import BaseMoeRoutingMethod


@torch.compile(options={"max-autotune": True})
def gen_fc2_alpha_fused(
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    alpha: Optional[torch.Tensor],
    alpha_max: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate fc2 alpha values, optionally normalized for FC1 alpha_post fusion.

    Instead of:
        1. zeros() -> scatter_() -> multiply with alpha (operates on large [N, E] tensor)

    We do:
        1. Gather alpha values for selected experts (small [N, top_k] tensor)
        2. Multiply scales with gathered alpha (small tensor operation)
        3. Optionally normalize by alpha_max for FC1 alpha_post fusion
        4. Scatter to output (single write to large tensor)

    This reduces memory bandwidth by avoiding read-modify-write on the large tensor.

    Args:
        token_selected_experts: Expert indices for each token [num_tokens, top_k]
        token_final_scales: Final scaling factors [num_tokens, top_k]
        alpha: Per-expert alpha values [expert_size]
        alpha_max: Max alpha value for normalization (optional)
        output: Pre-allocated output buffer [num_tokens, expert_size] (optional).
                If None, a new tensor will be allocated (not compatible with CUDA graph).
    """
    # Pre-compute scaled values on small tensor [num_tokens, top_k]
    if alpha is not None:
        # Gather alpha for selected experts: alpha[expert_idx] for each selection
        gathered_alpha = alpha[token_selected_experts.long()]  # [num_tokens, top_k]
        scaled_values = token_final_scales * gathered_alpha
    else:
        scaled_values = token_final_scales

    # Normalize by alpha_max for FC1 alpha_post fusion
    if alpha_max is not None:
        scaled_values = scaled_values / alpha_max

    # Use pre-allocated output or create new tensor
    if output is not None:
        output.zero_()
        fc2_alpha = output
    else:
        assert alpha is not None, (
            "alpha must be provided when output buffer is not pre-allocated, "
            "since expert_size cannot be inferred from token_final_scales alone"
        )
        num_tokens = token_selected_experts.shape[0]
        expert_size = alpha.shape[0]
        fc2_alpha = torch.zeros(
            [num_tokens, expert_size],
            dtype=torch.float32,
            device=token_selected_experts.device,
        )

    return fc2_alpha.scatter_(1, token_selected_experts.long(), scaled_values)


class DenseGEMMFusedMoE(CutlassFusedMoE):
    """CuteDSL flow of fused mixture of experts (MoE) Layer.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    # Memory buffer pool for CUDA graph compatibility
    buffers = get_memory_buffers()

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
    ):
        # FC2 DenseGEMM kernel tiles K dimension with MMA tile size 256.
        # weight_per_expert (= intermediate_size) must be 256-aligned so that
        # expert boundaries align with MMA tile boundaries.
        _MMA_TILE_K = 256
        assert intermediate_size % _MMA_TILE_K == 0, (
            f"DenseGEMMFusedMoE requires intermediate_size to be a multiple of "
            f"{_MMA_TILE_K} (got intermediate_size={intermediate_size}). "
            f"FC2 kernel cannot correctly split alpha_scale at expert boundaries "
            f"when weight_per_expert is not MMA tile-K aligned."
        )

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            without_comm=without_comm,
        )
        # Environment variable to control fc2_alpha fusion into FC1's alpha_post
        # Default: enabled (1). Set to "0" to use the original per-token per-expert fc2_alpha in FC2.
        self.use_fused_fc2_alpha = os.environ.get("TRTLLM_MOE_FUSED_FC2_ALPHA", "1") == "1"

        # Pre-register fc2_alpha_max buffer for fused fc2_alpha optimization.
        # Populated in load_weights() with max(fc2_alpha).
        self.register_buffer("fc2_alpha_max", torch.zeros(1, dtype=torch.float32))

        # Initialize auxiliary stream and events for gen_fc2_alpha_fused overlap with fc1
        if self.aux_stream_dict is None:
            self.aux_stream_dict = aux_stream_dict if aux_stream_dict is not None else {}
        if AuxStreamType.MoeFc2Alpha not in self.aux_stream_dict:
            self.aux_stream_dict[AuxStreamType.MoeFc2Alpha] = torch.cuda.Stream()
        if self.event_dict is None:
            self.event_dict = {}
        for key in [EventType.Main, EventType.MoeFc2Alpha]:
            if key not in self.event_dict:
                self.event_dict[key] = torch.cuda.Event()

    def load_weights(self, weights: List[Dict]):
        super().load_weights(weights)
        # Transpose w2_weight layout: (E, H, ...) -> (H, E, ...) for dense GEMM.
        # Use contiguous() to materialize the transposed layout, then copy back
        # into self.w2_weight in-place. This avoids clone() which would double
        # peak memory for large expert counts.
        w2_transposed = self.w2_weight.transpose(0, 1).contiguous()
        self.w2_weight.reshape([-1]).copy_(w2_transposed.reshape([-1]), non_blocking=True)
        del w2_transposed
        if self.has_any_quant:
            if self.has_nvfp4:
                self._transform_w2_weight_scale_for_min_latency()
                # Compute fc2_alpha_max for fused fc2_alpha optimization
                self.fc2_alpha_max.copy_(torch.max(self.fc2_alpha).reshape(1), non_blocking=True)
            else:
                raise ValueError(
                    f"{self.__class__.__name__} only supports nvfp4 quantization, "
                    f"got {self.quant_config.quant_mode}."
                )

    def _transform_w2_weight_scale_for_min_latency(self):
        """Transform w2_weight_scale for minimum latency path optimization."""
        # Calculate padded dimensions
        nrows = fp4_utils.pad_up(self.hidden_size, 128)
        ncols = fp4_utils.pad_up(
            self.intermediate_size_per_partition // self.scaling_vector_size, 4
        )

        # Clone and convert weight scale to uint8
        w2_weight_scale = self.w2_weight_scale.clone().view(torch.uint8)

        # Unswizzle the scale factor
        w2_weight_scale = unswizzle_sf(
            w2_weight_scale,
            self.hidden_size * self.expert_size_per_partition,
            self.intermediate_size_per_partition,
        )

        # Reshape and transpose for min latency layout
        w2_weight_scale = w2_weight_scale.reshape([self.expert_size_per_partition, nrows, ncols])
        w2_weight_scale = w2_weight_scale.transpose(0, 1).reshape(
            nrows, self.expert_size_per_partition * ncols
        )

        # Swizzle back with new layout
        w2_weight_scale = swizzle_sf(
            w2_weight_scale,
            self.hidden_size,
            self.expert_size_per_partition * self.intermediate_size_per_partition,
        )

        # Copy back to original tensor
        self.w2_weight_scale.copy_(
            w2_weight_scale.view(self.w2_weight_scale.dtype).view(self.w2_weight_scale.shape),
            non_blocking=True,
        )

    def select_alltoall_method_type(self) -> AlltoallMethodType:
        return AlltoallMethodType.NotEnabled

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True
        ):
            if self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CuteDslFusedMoEMethod()
        return super()._get_quant_method()

    def quantize_input(
        self, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
    ):
        """Quantize inputs prior to post-communication (alltoall/allgather) or before MoE computation.

        Args:
            x: Input tensor to quantize
            post_quant_comm:
                If True, quantize for post-quant communication path.
                If False, quantize for non-communication path

        Returns: (x, x_sf) where x_sf is already reshaped to 2D if needed

        For quantization methods that produce scaling factors:
        - x_sf is reshaped from 1D to 2D: [num_elements] -> [batch_size, ceil_div(hidden_size, scaling_vector_size)]
        - The 2D shape is required for proper handling in alltoall/allgather operations
        - scaling_vector_size is typically the group size for block-wise quantization
        """
        x_sf = None
        if self.has_nvfp4:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, (
                    "Fp4QuantizedTensor should not be swizzled before communication"
                )
                x_row = x.shape[0]
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                x_row = x.shape[0]
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False, False
                )
        else:
            raise ValueError(
                f"{self.__class__.__name__} only supports nvfp4 quantization, "
                f"got {self.quant_config.quant_mode}."
            )

        if x_sf is not None:
            x_sf = x_sf.view(x_row, -1)

        return x, x_sf

    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        """Run MoE computation with NVFP4 quantization.

        Args:
            x: Input tensor
            token_selected_experts: Expert indices for each token
            token_final_scales: Final scaling factors for each token
            x_sf: Input scale factors
            enable_alltoall: Whether alltoall communication is enabled

        Note:
            The implementation is controlled by TRTLLM_MOE_FUSED_FC2_ALPHA env var (default: enabled).
            When enabled, fc2_alpha is fused into FC1's alpha_post with scalar fc2_alpha_max in FC2.
            When disabled, uses the original per-token per-expert fc2_alpha in FC2.
        """
        assert self.has_nvfp4
        num_tokens = x.shape[0]

        # Get pre-allocated buffer for fc2_alpha (CUDA graph compatible)
        capture_graph = torch.cuda.is_current_stream_capturing()
        fc2_alpha_buffer = DenseGEMMFusedMoE.buffers.get_buffer(
            (num_tokens, self.expert_size_per_partition),
            dtype=torch.float32,
            buffer_name="fc2_alpha",
            reserve_buffer=capture_graph,
        )

        if self.use_fused_fc2_alpha:
            # New implementation: fuse fc2_alpha into FC1's alpha_post
            x_sf = swizzle_sf(x_sf, num_tokens, self.hidden_size)

            # Generate normalized fc2_alpha for FC1 alpha_post fusion
            fc2_alpha_normalized = gen_fc2_alpha_fused(
                token_selected_experts,
                token_final_scales,
                self.fc2_alpha,
                self.fc2_alpha_max,  # Normalize by max for FC1 alpha_post
                fc2_alpha_buffer,  # Pre-allocated buffer
            )

            # FC1: GEMM + SwiGLU with post-SwiGLU alpha scaling (fused fc2_alpha)
            fc1_output, fc1_output_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
                x,
                self.w3_w1_weight.view(torch.uint8),
                x_sf,
                self.w3_w1_weight_scale,
                self.fc31_alpha,
                fc2_alpha_normalized,  # Pass normalized fc2_alpha as alpha_post
                self.fc2_input_scale,
                expert_count=self.expert_size_per_partition,
                weight_per_expert=2 * self.intermediate_size_per_partition,
                output_dtype=torch.float4_e2m1fn_x2,
                scaling_vector_size=self.scaling_vector_size,
            )

            # FC2: Standard nvfp4_gemm with scalar alpha = fc2_alpha_max
            final_hidden_states = torch.ops.trtllm.nvfp4_gemm(
                fc1_output.view(torch.uint8),
                self.w2_weight.view(torch.uint8).reshape(self.hidden_size, -1),
                fc1_output_sf.view(torch.uint8).reshape(-1),
                self.w2_weight_scale.view(torch.uint8),
                self.fc2_alpha_max,
                torch.bfloat16,
                to_userbuffers=False,
                allowed_backends="cutlass,cublaslt,cutedsl,cuda_core",
            )
        else:
            # Original implementation: per-token per-expert fc2_alpha in FC2
            self.event_dict[EventType.Main].record()
            x_sf = swizzle_sf(x_sf, num_tokens, self.hidden_size)

            # FC1: GEMM + SwiGLU, output is fp4 quantized
            fc1_output, fc1_output_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
                x,
                self.w3_w1_weight.view(torch.uint8),
                x_sf,
                self.w3_w1_weight_scale,
                self.fc31_alpha,
                None,  # alpha_post: no post-SwiGLU scaling
                self.fc2_input_scale,
                expert_count=self.expert_size_per_partition,
                weight_per_expert=2 * self.intermediate_size_per_partition,
                output_dtype=torch.float4_e2m1fn_x2,
                scaling_vector_size=self.scaling_vector_size,
            )

            with torch.cuda.stream(self.aux_stream_dict[AuxStreamType.MoeFc2Alpha]):
                self.event_dict[EventType.Main].wait()
                fc2_alpha = gen_fc2_alpha_fused(
                    token_selected_experts,
                    token_final_scales,
                    self.fc2_alpha,
                    output=fc2_alpha_buffer,  # Use pre-allocated buffer
                )
                self.event_dict[EventType.MoeFc2Alpha].record()

            self.event_dict[EventType.MoeFc2Alpha].wait()

            # FC2: input k = expert_count * intermediate_size (after SwiGLU)
            final_hidden_states = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_fc2_blackwell(
                fc1_output,
                self.w2_weight.view(torch.uint8).reshape(self.hidden_size, -1),
                fc1_output_sf.reshape(-1),
                self.w2_weight_scale,
                fc2_alpha,
                expert_count=self.expert_size_per_partition,
                weight_per_expert=self.intermediate_size_per_partition,
                output_dtype=torch.bfloat16,
                scaling_vector_size=self.scaling_vector_size,
            )

        return final_hidden_states

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run MoE computation with DenseGEMM backend (NVFP4 only).

        Args:
            x: Input hidden states (pre-quantized to NVFP4)
            token_selected_experts: Expert IDs [num_tokens, top_k]. If EPLB is enabled,
                                    this represents expert slots [num_tokens, top_k] instead.
            token_final_scales: Final scaling factors for each token
            x_sf: Input scale factors for NVFP4
            enable_alltoall: Whether alltoall communication is enabled.
            **kwargs: Additional arguments for forward compatibility.

        Returns:
            final_hidden_states tensor.
        """
        assert self.has_nvfp4, (
            f"{self.__class__.__name__} only supports nvfp4 quantization, "
            f"got {self.quant_config.quant_mode}."
        )
        return self.run_moe_nvfp4(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            enable_alltoall=enable_alltoall,
        )

    def forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        # Currently, the default path is that ConfigurableMoE calls DenseGEMMFusedMoE.run_moe.
        # This forward_chunk method is a reference implementation of the legacy path.
        # Apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
        assert token_selected_experts.shape[1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        x, x_sf = self.quantize_input(x)

        if self.use_dp and self.parallel_size > 1:
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens,
            )

        x = self.run_moe(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            enable_alltoall=False,
        )
        return x
