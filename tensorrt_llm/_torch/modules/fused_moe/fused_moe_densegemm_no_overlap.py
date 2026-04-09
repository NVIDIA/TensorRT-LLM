# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.quantization.utils import fp4_utils

from ...distributed import allgather
from ...memory_buffer_utils import get_memory_buffers
from ...model_config import ModelConfig
from ...utils import AuxStreamType, Fp4QuantizedTensor, swizzle_sf, unswizzle_sf
from .interface import MoE, MoEWeightLoadingMode
from .quantization import NVFP4CuteDslFusedMoEMethod
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


class NoOverlapDenseGEMMFusedMoE(MoE):
    """Single-stream, no-SM-partition DenseGEMM MoE for performance baseline.

    This backend uses CuTe DSL dense GEMM kernels with fused SwiGLU for MoE
    computation. It supports NVFP4 quantization only and is restricted to
    SM100/SM103 (Blackwell) architectures.

    Unlike the gc/smp variants, all ops run sequentially on a single CUDA stream.
    No max_active_clusters or sm_budget limit is applied to any kernel (GPU uses
    all available SMs). Use this as a baseline to measure the benefit of SM
    partitioning + stream overlap in DenseGEMMFusedMoE.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams (unused).
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    # Memory buffer pool for CUDA graph compatibility
    buffers = get_memory_buffers()

    # DenseGEMM only supports SM100 and SM103 (Blackwell CuTe DSL kernels).
    _SUPPORTED_SM_VERSIONS = (100, 103)

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> tuple:
        """Check if NoOverlapDenseGEMMFusedMoE can implement the given configuration."""
        from tensorrt_llm._utils import get_sm_version

        from .interface import _warn_and_return

        sm_version = get_sm_version()
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            return _warn_and_return(
                f"NoOverlapDenseGEMMFusedMoE requires SM {cls._SUPPORTED_SM_VERSIONS}, got SM{sm_version}"
            )

        if quant_algo != QuantAlgo.NVFP4:
            return _warn_and_return(
                f"NoOverlapDenseGEMMFusedMoE only supports NVFP4 quantization (got quant_algo={quant_algo})"
            )

        if swiglu_gptoss_style:
            return _warn_and_return(
                "NoOverlapDenseGEMMFusedMoE does not support swiglu_gptoss_style"
            )

        return (True, None)

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
        activation_type=None,
    ):
        # DenseGEMM CuTe DSL kernels only support SM100 and SM103.
        from tensorrt_llm._utils import get_sm_version

        from ...utils import ActivationType

        sm_version = get_sm_version()
        assert sm_version in self._SUPPORTED_SM_VERSIONS, (
            f"NoOverlapDenseGEMMFusedMoE only supports SM {self._SUPPORTED_SM_VERSIONS} "
            f"(got SM {sm_version}). The CuTe DSL kernels require Blackwell architecture."
        )

        # DenseGEMM kernel hardcodes SwiGLU fusion — reject other activation types.
        if activation_type is None:
            activation_type = ActivationType.Swiglu
        assert activation_type == ActivationType.Swiglu, (
            f"NoOverlapDenseGEMMFusedMoE only supports SwiGLU activation "
            f"(got activation_type={activation_type}). "
            f"The FC1 kernel fuses SwiGLU into the GEMM epilogue."
        )

        # FC2 DenseGEMM kernel tiles K dimension with MMA tile size 256.
        _MMA_TILE_K = 256
        assert intermediate_size % _MMA_TILE_K == 0, (
            f"NoOverlapDenseGEMMFusedMoE requires intermediate_size to be a multiple of "
            f"{_MMA_TILE_K} (got intermediate_size={intermediate_size}). "
            f"FC2 kernel cannot correctly split alpha_scale at expert boundaries "
            f"when weight_per_expert is not MMA tile-K aligned."
        )

        # Call MoE base class directly (not CutlassFusedMoE).
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
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
        )

        # Environment variable to control fc2_alpha fusion into FC1's alpha_post.
        # Default: disabled (0). Set to "1" to enable fusion (known accuracy issue under TP).
        self.use_fused_fc2_alpha = os.environ.get("TRTLLM_MOE_FUSED_FC2_ALPHA", "0") == "1"

        # Pre-register fc2_alpha_max buffer for fused fc2_alpha optimization.
        # Populated in load_weights() with max(fc2_alpha).
        self.register_buffer("fc2_alpha_max", torch.zeros(1, dtype=torch.float32))

        # No SM budget computation — all SMs used for all kernels.
        # No auxiliary streams needed — sequential single-stream execution.
        if self.aux_stream_dict is None:
            self.aux_stream_dict = aux_stream_dict if aux_stream_dict is not None else {}
        self.event_dict = {}

        # Weight creation
        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _supports_load_balancer(self) -> bool:
        """NoOverlapDenseGEMMFusedMoE supports load balancer."""
        return True

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True
        ):
            if self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CuteDslFusedMoEMethod()
            raise ValueError(
                f"{self.__class__.__name__} only supports NVFP4 quantization, "
                f"got {self.quant_config.quant_mode}."
            )
        raise ValueError(
            f"{self.__class__.__name__} requires quantization (NVFP4), "
            f"but no quantization config was provided."
        )

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        kargs = {}
        if "allow_partial_loading" in inspect.getfullargspec(self.quant_method.load_weights).args:
            kargs["allow_partial_loading"] = allow_partial_loading
        self.quant_method.load_weights(self, weights, self.weight_loading_mode, **kargs)

        # Transpose w2_weight layout: (E, H, ...) -> (H, E, ...) for dense GEMM.
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

    def post_load_weights(self):
        self.quant_method.post_load_weights(self)

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

    def quantize_input(
        self, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
    ):
        """Quantize inputs prior to post-communication (alltoall/allgather) or before MoE computation."""
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
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        router_weight_t: Optional[torch.Tensor] = None,
        router_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-stream MoE forward, no SM partitioning.

        Sequential execution order on main stream:
            Router GEMM -> routing -> (gen_fc2_alpha | swizzle_sf) -> FC1 -> FC2

        All kernels run without SM budget constraints (GPU uses all available SMs).

        Args:
            x: Input tensor
            token_selected_experts: Must be None (routing done internally)
            token_final_scales: Must be None (routing done internally)
            x_sf: Input scale factors
            enable_alltoall: Whether alltoall communication is enabled
            router_weight_t: Router weight matrix [hidden, num_experts] (transposed)
            router_input: Optional separate router input; defaults to x if None
        """
        assert self.has_nvfp4
        assert token_selected_experts is None and token_final_scales is None, (
            "NoOverlapDenseGEMMFusedMoE only supports internal routing path. "
            "Expected token_selected_experts/token_final_scales to be None."
        )
        assert router_weight_t is not None, (
            "NoOverlapDenseGEMMFusedMoE internal routing requires router_weight_t."
        )
        if router_input is None:
            router_input = x
        assert router_input.ndim == 2 and router_weight_t.ndim == 2, (
            "router_input and router_weight_t must both be rank-2 tensors."
        )
        assert router_input.shape[1] == router_weight_t.shape[0], (
            "NoOverlapDenseGEMMFusedMoE internal routing shape mismatch: "
            f"router_input.shape={tuple(router_input.shape)}, "
            f"router_weight_t.shape={tuple(router_weight_t.shape)}"
        )

        num_tokens = x.shape[0]

        # Get pre-allocated buffer for fc2_alpha (CUDA graph compatible)
        capture_graph = torch.cuda.is_current_stream_capturing()
        fc2_alpha_buffer = NoOverlapDenseGEMMFusedMoE.buffers.get_buffer(
            (num_tokens, self.expert_size_per_partition),
            dtype=torch.float32,
            buffer_name="fc2_alpha_no_overlap",
            reserve_buffer=capture_graph,
        )

        # Router GEMM (all SMs, single stream, no SM budget)
        router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
            router_input,
            router_weight_t,
            None,
            torch.float32,
        )

        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
        token_selected_experts = token_selected_experts.to(torch.int32)

        # Append fused shared experts (always active, scale=1.0) if any.
        num_routing_experts = router_weight_t.shape[1]
        if self.expert_size_per_partition > num_routing_experts:
            n_shared = self.expert_size_per_partition - num_routing_experts
            M_tok = router_input.shape[0]
            shared_ids = (
                torch.arange(
                    num_routing_experts,
                    self.expert_size_per_partition,
                    dtype=torch.int32,
                    device=x.device,
                )
                .unsqueeze(0)
                .expand(M_tok, n_shared)
            )
            shared_scales = torch.ones(M_tok, n_shared, dtype=torch.float32, device=x.device)
            token_selected_experts = torch.cat([token_selected_experts, shared_ids], dim=1)
            token_final_scales = torch.cat([token_final_scales, shared_scales], dim=1)

        x_sf = swizzle_sf(x_sf, num_tokens, self.hidden_size)

        if self.use_fused_fc2_alpha:
            # Fused path: fc2_alpha fused into FC1's alpha_post
            fc2_alpha_normalized = gen_fc2_alpha_fused(
                token_selected_experts,
                token_final_scales,
                self.fc2_alpha,
                self.fc2_alpha_max,  # Normalize by max for FC1 alpha_post
                fc2_alpha_buffer,  # Pre-allocated buffer
            )

            # FC1: GEMM + SwiGLU with post-SwiGLU alpha scaling (fused fc2_alpha), no SM limit
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
            # Sequential path: gen_fc2_alpha then FC1 then FC2, all without SM budget
            fc2_alpha = gen_fc2_alpha_fused(
                token_selected_experts,
                token_final_scales,
                self.fc2_alpha,
                output=fc2_alpha_buffer,  # Use pre-allocated buffer
            )

            # FC1: GEMM + SwiGLU, output is fp4 quantized, no SM limit
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
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run MoE computation with DenseGEMM backend (NVFP4 only), single-stream no-overlap variant.

        Args:
            x: Input hidden states (pre-quantized to NVFP4)
            token_selected_experts: Must be None (routing done internally via router_weight_t)
            token_final_scales: Must be None (routing done internally)
            x_sf: Input scale factors for NVFP4
            enable_alltoall: Whether alltoall communication is enabled.
            **kwargs: Must contain 'router_weight_t'; optionally 'router_input'.

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
            router_weight_t=kwargs.get("router_weight_t"),
            router_input=kwargs.get("router_input"),
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
        # Currently, the default path is that ConfigurableMoE calls run_moe.
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

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert do_finalize, "NoOverlapDenseGEMMFusedMoE does not support do_finalize=False"

        is_first_call = self.repeat_idx == 0
        is_last_call = self.repeat_idx == self.repeat_count - 1

        outputs = self.forward_chunk(
            x,
            router_logits,
            output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            repeating_info=(is_first_call, is_last_call),
        )
        outputs = self.reducescatter_or_allreduce(
            outputs,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        if self.use_dp and self.parallel_size > 1:
            rank = self.parallel_rank
            outputs = outputs[: all_rank_num_tokens[rank]]
        self.repeat_idx = 0 if self.repeat_idx == self.repeat_count - 1 else self.repeat_idx + 1
        return outputs
