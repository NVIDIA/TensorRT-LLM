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

"""
MoE Op Backend Registry for TRTLLMGenFusedMoE.

This module provides a registry-based backend abstraction for different MoE implementations
(flashinfer and trtllm), reducing code duplication and improving maintainability.
"""

import os
from typing import Dict, List, Optional, Tuple, Type

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from ...utils import ActType_TrtllmGen

# Global registry for MoE backends
_MOE_OP_BACKEND_REGISTRY: Dict[str, Type["MoEOpBackend"]] = {}


if triton is not None:

    @triton.jit
    def pack_topk_ids_kernel(
        expert_ids_ptr,
        expert_weights_ptr,
        output_ptr,
        local_expert_offset,
        stride_ids_row,
        stride_ids_col,
        stride_weights_row,
        stride_weights_col,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_idx = linear_idx // n_cols
        col_idx = linear_idx % n_cols

        mask = linear_idx < (n_rows * n_cols)

        ids_offset = row_idx * stride_ids_row + col_idx * stride_ids_col
        weights_offset = row_idx * stride_weights_row + col_idx * stride_weights_col

        expert_ids = tl.load(expert_ids_ptr + ids_offset, mask=mask)
        local_ids = expert_ids - local_expert_offset

        expert_weights = tl.load(expert_weights_ptr + weights_offset, mask=mask)
        expert_weights_bf16 = expert_weights.to(tl.bfloat16)
        expert_weights_int16 = expert_weights_bf16.to(tl.int16, bitcast=True)
        expert_weights_int32 = expert_weights_int16.to(tl.int32) & 0xFFFF

        packed_topk_ids = (local_ids.to(tl.int32) << 16) | expert_weights_int32
        tl.store(output_ptr + linear_idx, packed_topk_ids, mask=mask)


def pack_topk_ids(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack expert ids and routing weights into TRTLLM-Gen's routed MoE format."""
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("Expected 2D top-k ids and weights tensors.")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"Mismatched top-k ids and weights shapes: {topk_ids.shape} vs {topk_weights.shape}."
        )

    if output is None:
        output = torch.empty(topk_ids.shape, dtype=torch.int32, device=topk_ids.device)

    # Fallback to CPU just in case
    if triton is None or not topk_ids.is_cuda or not topk_weights.is_cuda:
        packed_topk_ids = ((topk_ids.to(torch.int32) - local_expert_offset) << 16) | (
            topk_weights.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        )
        output.copy_(packed_topk_ids)
        return output

    n_rows, n_cols = topk_ids.shape
    n_elements = n_rows * n_cols
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)

    pack_topk_ids_kernel[grid](
        topk_ids,
        topk_weights,
        output,
        local_expert_offset,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        n_rows,
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return output


def register_op_backend(name: str):
    """Decorator to register a MoE op backend class."""

    def decorator(cls: Type["MoEOpBackend"]) -> Type["MoEOpBackend"]:
        _MOE_OP_BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_op_backend(name: str) -> "MoEOpBackend":
    """Get a registered backend instance by name."""
    if name not in _MOE_OP_BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown op backend '{name}'. Available: {list(_MOE_OP_BACKEND_REGISTRY.keys())}"
        )
    return _MOE_OP_BACKEND_REGISTRY[name]()


def get_available_op_backend() -> "MoEOpBackend":
    """Get the best available backend (prefer flashinfer if available)."""
    if "flashinfer" in _MOE_OP_BACKEND_REGISTRY:
        try:
            return get_op_backend("flashinfer")
        except ImportError:
            pass
    return get_op_backend("trtllm")


class MoEOpBackend:
    """
    Base class for MoE Op backend operations.

    All backend-specific operations are accessed through this unified interface.
    Subclasses register themselves using the @register_op_backend decorator.
    """

    # ==================== Quantization Operations ====================

    def fp4_quantize(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP4 format."""
        raise NotImplementedError

    def mxfp8_quantize(
        self,
        input: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
        alignment: int = 32,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to MXFP8 format."""
        raise NotImplementedError

    # ==================== MoE Runner Operations ====================

    def run_fp8_block_scale_moe(
        self,
        router_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        gated_act_type: int = 0,
        output: Optional[torch.Tensor] = None,
        use_shuffled_weight: bool = False,
        weight_layout: int = 0,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
    ) -> torch.Tensor:
        """Run FP8 block scale MoE computation."""
        raise NotImplementedError

    def run_fp4_block_scale_moe(
        self,
        router_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: Optional[torch.Tensor],
        output1_scale_gate_scalar: Optional[torch.Tensor],
        output2_scale_scalar: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        topk_weights: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        valid_hidden_size: Optional[int] = None,
        valid_intermediate_size: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        gated_act_type: int = 0,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
    ) -> List[torch.Tensor]:
        """Run FP4 block scale MoE computation."""
        raise NotImplementedError

    def run_bf16_moe(
        self,
        router_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        gated_act_type: int = 0,
        output: Optional[torch.Tensor] = None,
        use_shuffled_weight: bool = False,
        weight_layout: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
    ) -> torch.Tensor:
        """Run BF16 MoE computation."""
        raise NotImplementedError


# ==================== TRTLLM Backend ====================
@register_op_backend("trtllm")
class TRTLLMOpBackend(MoEOpBackend):
    """TRTLLM native op backend implementation."""

    def __init__(self):
        from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe
        from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll

        self._MnnvlMemory = MnnvlMemory
        self._MnnvlMoe = MnnvlMoe
        self._MoeAlltoAll = MoeAlltoAll

    # Quantization
    def fp4_quantize(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ):
        return torch.ops.trtllm.fp4_quantize(
            input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )

    def mxfp8_quantize(
        self,
        input: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
        alignment: int = 32,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.mxfp8_quantize(input, is_sf_swizzled_layout, alignment=alignment)

    # MoE Runners
    def run_fp8_block_scale_moe(
        self,
        router_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        topk_weights=None,
        topk_ids=None,
        gated_act_type=0,
        output=None,
        use_shuffled_weight=False,
        weight_layout=0,
        enable_pdl=None,
        tune_max_num_tokens=8192,
    ):
        return torch.ops.trtllm.fp8_block_scale_moe_runner(
            router_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            act_type=gated_act_type,
            output=output,
        )

    def run_fp4_block_scale_moe(
        self,
        router_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize=True,
        topk_weights=None,
        topk_ids=None,
        valid_hidden_size=None,
        valid_intermediate_size=None,
        enable_pdl=None,
        gated_act_type=0,
        output=None,
        tune_max_num_tokens=8192,
    ):
        hidden_size = gemm1_weights.shape[-1] * 2
        if hidden_states.dtype == torch.uint8 or hidden_states.dtype == torch.float8_e4m3fn:
            if (
                gemm1_weights_scale is not None
                and gemm1_weights_scale.shape[-1] == hidden_size // 16
            ):
                # nvfp4
                outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
                    router_logits,
                    routing_bias,
                    hidden_states,
                    hidden_states_scale.view(torch.float8_e4m3fn)
                    if hidden_states_scale is not None
                    else None,
                    gemm1_weights,
                    gemm1_weights_scale.view(torch.float8_e4m3fn),
                    gemm1_bias,
                    gemm1_alpha,
                    gemm1_beta,
                    gemm1_clamp_limit,
                    gemm2_weights,
                    gemm2_weights_scale.view(torch.float8_e4m3fn),
                    gemm2_bias,
                    output1_scale_scalar,
                    output1_scale_gate_scalar,
                    output2_scale_scalar,
                    num_experts,
                    top_k,
                    n_group,
                    topk_group,
                    intermediate_size,
                    local_expert_offset,
                    local_num_experts,
                    routed_scaling_factor,
                    routing_method_type,
                    do_finalize=do_finalize,
                    act_type=gated_act_type,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    output=output,
                )
                if not do_finalize:
                    return outputs
                else:
                    final_hidden_states = outputs[0]
                    return final_hidden_states
            elif (
                gemm1_weights_scale is not None
                and gemm1_weights_scale.shape[-1] == hidden_size // 32
            ):
                # mxfp4
                return torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
                    router_logits,
                    routing_bias,
                    hidden_states,
                    hidden_states_scale,
                    gemm1_weights,
                    gemm1_weights_scale,
                    gemm1_bias,
                    gemm1_alpha,
                    gemm1_beta,
                    gemm1_clamp_limit,
                    gemm2_weights,
                    gemm2_weights_scale,
                    gemm2_bias,
                    num_experts,
                    top_k,
                    n_group,
                    topk_group,
                    intermediate_size,
                    valid_hidden_size,
                    valid_intermediate_size,
                    local_expert_offset,
                    local_num_experts,
                    routed_scaling_factor,
                    routing_method_type,
                    gated_act_type,
                    topk_weights,
                    topk_ids,
                    output=output,
                )

        elif hidden_states.dtype == torch.bfloat16:
            return torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner(
                router_logits,
                routing_bias,
                hidden_states,
                gemm1_weights,
                gemm1_weights_scale,
                gemm1_bias,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
                gemm2_weights,
                gemm2_weights_scale,
                gemm2_bias,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                valid_hidden_size,
                valid_intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                routing_method_type,
                act_type=gated_act_type,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=output,
            )

    def run_bf16_moe(
        self,
        router_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        topk_weights=None,
        topk_ids=None,
        gated_act_type=0,
        output=None,
        use_shuffled_weight=False,
        weight_layout=0,
        do_finalize=True,
        enable_pdl=None,
        tune_max_num_tokens=8192,
    ):
        raise NotImplementedError(
            "TRTLLM native op backend does not support unquantized BF16 TRTLLM-Gen fused MoE. "
            "Enable FlashInfer fused MoE for TRTLLM backend."
        )


# ==================== Flashinfer Backend ====================
@register_op_backend("flashinfer")
class FlashinferOpBackend(MoEOpBackend):
    """Flashinfer op backend implementation."""

    def __init__(self):
        import flashinfer.fused_moe as _flashinfer_fused_moe
        from flashinfer.fp4_quantization import fp4_quantize as _flashinfer_fp4_quantize
        from flashinfer.fp8_quantization import mxfp8_quantize as _flashinfer_mxfp8_quantize
        from flashinfer.fused_moe.core import ActivationType as _flashinfer_activation_type
        from flashinfer.fused_moe.core import RoutingMethodType as _flashinfer_routing_method_type

        from ..fused_moe.routing import RoutingMethodType as _trtllmgen_routing_method_type

        self._trtllmgen_routing_method_type = _trtllmgen_routing_method_type

        self._activation_type = _flashinfer_activation_type
        self._routing_method_type = _flashinfer_routing_method_type
        self._fused_moe = _flashinfer_fused_moe
        self._fp4_quantize = _flashinfer_fp4_quantize
        self._mxfp8_quantize = _flashinfer_mxfp8_quantize

        # need to add this to the flashinfer side
        os.environ["FLASHINFER_EXTRA_LDFLAGS"] = "-Wl,-Bsymbolic-functions"

    def cvt_activation_type(self, activation_type) -> int:
        """Convert TRT-LLM ActivationType to FlashInfer ActivationType int value."""
        _flashinfer = self._activation_type
        _mapping = {
            0: _flashinfer.Swiglu.value,
            1: _flashinfer.Relu2.value,
            2: _flashinfer.Silu.value,
        }
        if activation_type not in _mapping:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        return int(_mapping[activation_type])

    def cvt_routing_method_type(self, routing_method_type) -> int:
        """Convert TRT-LLM RoutingMethodType to FlashInfer RoutingMethodType int value."""
        _trtllm = self._trtllmgen_routing_method_type
        _flashinfer = self._routing_method_type
        _mapping = {
            _trtllm.Default: _flashinfer.Default,
            _trtllm.Renormalize: _flashinfer.Renormalize,
            _trtllm.DeepSeekV3: _flashinfer.DeepSeekV3,
            _trtllm.Llama4: _flashinfer.Llama4,
            _trtllm.RenormalizeNaive: _flashinfer.RenormalizeNaive,
            _trtllm.Unspecified: _flashinfer.Unspecified,
        }
        if routing_method_type not in _mapping:
            raise ValueError(f"Unsupported routing method type: {routing_method_type}")
        return int(_mapping[routing_method_type])

    # Quantization
    def fp4_quantize(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ):
        return self._fp4_quantize(
            input,
            global_scale,
            sf_vec_size,
            sf_use_ue8m0,
            is_sf_swizzled_layout,
            is_sf_8x4_layout,
            enable_pdl,
        )

    def mxfp8_quantize(
        self,
        input: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
        alignment: int = 32,
        enable_pdl: Optional[bool] = None,
    ):
        return self._mxfp8_quantize(
            input, is_sf_swizzled_layout, alignment=alignment, enable_pdl=enable_pdl
        )

    # MoE Runners
    def run_fp8_block_scale_moe(
        self,
        router_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        topk_weights=None,
        topk_ids=None,
        gated_act_type=0,
        output=None,
        use_shuffled_weight=False,
        weight_layout=0,
        enable_pdl=None,
        tune_max_num_tokens=8192,
    ):
        if router_logits is not None:
            return self._fused_moe.trtllm_fp8_block_scale_moe(
                router_logits,
                routing_bias,
                hidden_states,
                hidden_states_scale,
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                self.cvt_routing_method_type(routing_method_type),
                use_shuffled_weight=use_shuffled_weight,
                weight_layout=weight_layout,
                enable_pdl=enable_pdl,
                tune_max_num_tokens=tune_max_num_tokens,
            )
        else:
            packed_topk_ids = pack_topk_ids(topk_ids, topk_weights, local_expert_offset)
            # Run with pre-computed routing (packed format)
            return self._fused_moe.trtllm_fp8_block_scale_routed_moe(
                topk_ids=packed_topk_ids,
                routing_bias=routing_bias,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=n_group,
                topk_group=topk_group,
                intermediate_size=intermediate_size,
                local_expert_offset=local_expert_offset,
                local_num_experts=num_experts,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=self.cvt_routing_method_type(routing_method_type),
                use_shuffled_weight=use_shuffled_weight,
                weight_layout=weight_layout,
                enable_pdl=enable_pdl,
                output=output,
                tune_max_num_tokens=tune_max_num_tokens,
            )

    def run_fp4_block_scale_moe(
        self,
        router_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize=True,
        topk_weights=None,
        topk_ids=None,
        valid_hidden_size=None,
        valid_intermediate_size=None,
        enable_pdl=None,
        gated_act_type=0,
        output=None,
        tune_max_num_tokens=8192,
    ):
        if router_logits is not None:
            outputs = self._fused_moe.trtllm_fp4_block_scale_moe(
                router_logits,
                routing_bias,
                hidden_states,
                hidden_states_scale.view(torch.float8_e4m3fn)
                if hidden_states_scale is not None
                else None,
                gemm1_weights,
                gemm1_weights_scale.view(torch.float8_e4m3fn),
                gemm1_bias,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
                gemm2_weights,
                gemm2_weights_scale.view(torch.float8_e4m3fn),
                gemm2_bias,
                output1_scale_scalar,
                output1_scale_gate_scalar,
                output2_scale_scalar,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                self.cvt_routing_method_type(routing_method_type),
                do_finalize=do_finalize,
                enable_pdl=enable_pdl,
                activation_type=self.cvt_activation_type(gated_act_type),
                output=output,
                tune_max_num_tokens=tune_max_num_tokens,
            )
        else:
            packed_tensor = pack_topk_ids(topk_ids, topk_weights, local_expert_offset)
            outputs = self._fused_moe.trtllm_fp4_block_scale_routed_moe(
                packed_tensor,
                routing_bias,
                hidden_states,
                hidden_states_scale.view(torch.float8_e4m3fn)
                if hidden_states_scale is not None
                else None,
                gemm1_weights,
                gemm1_weights_scale.view(torch.float8_e4m3fn),
                gemm1_bias,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
                gemm2_weights,
                gemm2_weights_scale.view(torch.float8_e4m3fn),
                gemm2_bias,
                output1_scale_scalar,
                output1_scale_gate_scalar,
                output2_scale_scalar,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                self.cvt_routing_method_type(routing_method_type),
                do_finalize=do_finalize,
                enable_pdl=enable_pdl,
                activation_type=self.cvt_activation_type(gated_act_type),
                output=output,
                tune_max_num_tokens=tune_max_num_tokens,
            )
        if not do_finalize:
            if outputs[2].dim() != 2:
                outputs[2] = outputs[2].view(-1, top_k)
            return outputs
        else:
            final_hidden_states = outputs[0]
            return final_hidden_states

    def run_bf16_moe(
        self,
        router_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        topk_weights=None,
        topk_ids=None,
        gated_act_type=0,
        output=None,
        use_shuffled_weight=False,
        weight_layout=0,
        do_finalize=True,
        enable_pdl=None,
        tune_max_num_tokens=8192,
    ):
        # FlashInfer BF16 MoE does not expose an activation_type argument.
        # TRTLLMGen constrains the BF16 path to Swiglu, so reject anything
        # else here instead of silently calling a mismatched kernel.
        if gated_act_type != ActType_TrtllmGen.SwiGlu:
            raise ValueError("FlashInfer BF16 fused MoE only supports Swiglu activation.")

        if router_logits is not None:
            result = self._fused_moe.trtllm_bf16_moe(
                routing_logits=router_logits,
                routing_bias=routing_bias,
                hidden_states=hidden_states,
                gemm1_weights=gemm1_weights,
                gemm2_weights=gemm2_weights,
                num_experts=num_experts,
                top_k=top_k,
                n_group=n_group,
                topk_group=topk_group,
                intermediate_size=intermediate_size,
                local_expert_offset=local_expert_offset,
                local_num_experts=local_num_experts,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=self.cvt_routing_method_type(routing_method_type),
                use_shuffled_weight=use_shuffled_weight,
                weight_layout=weight_layout,
                do_finalize=do_finalize,
                enable_pdl=enable_pdl,
                tune_max_num_tokens=tune_max_num_tokens,
            )
        else:
            packed_topk_ids = pack_topk_ids(topk_ids, topk_weights, local_expert_offset)
            result = self._fused_moe.trtllm_bf16_routed_moe(
                packed_topk_ids,
                hidden_states,
                gemm1_weights,
                gemm2_weights,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                self.cvt_routing_method_type(routing_method_type),
                use_shuffled_weight=use_shuffled_weight,
                weight_layout=weight_layout,
                do_finalize=do_finalize,
                enable_pdl=enable_pdl,
                tune_max_num_tokens=tune_max_num_tokens,
            )

        if output is not None and do_finalize:
            output.copy_(result)
            return output
        return result
