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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from tensorrt_llm.logger import logger

# Global registry for MoE backends
_MOE_OP_BACKEND_REGISTRY: Dict[str, Type["MoEOpBackend"]] = {}


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

    # ==================== MNNVL Memory Operations ====================

    def mnnvl_initialize(self) -> None:
        """Initialize MNNVL memory."""
        raise NotImplementedError

    def mnnvl_get_workspaces(self, mapping: Any) -> Any:
        """Get MoE workspaces for NVLink communication."""
        raise NotImplementedError

    def mnnvl_get_prepare_workspace(self, mapping: Any) -> Any:
        """Get MoE prepare workspace for NVLink communication."""
        raise NotImplementedError

    def mnnvl_moe_alltoallv_prepare_without_allgather(
        self,
        expert_ids: torch.Tensor,
        expert_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
        scales: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """Prepare for MNNVL MoE AlltoAllV."""
        raise NotImplementedError

    def mnnvl_moe_alltoallv(
        self,
        x: Union[torch.Tensor, List[Optional[torch.Tensor]]],
        alltoall_info,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Execute MNNVL MoE AlltoAllV."""
        raise NotImplementedError

    def mnnvl_moe_alltoallv_combine(
        self,
        x: torch.Tensor,
        alltoall_info,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
        use_low_precision_combine: bool = False,
        do_reduce: bool = True,
    ) -> torch.Tensor:
        """Combine results from MNNVL MoE AlltoAllV."""
        raise NotImplementedError

    # ==================== MoeAlltoAll Operations ====================

    def get_a2a_workspace_size(
        self,
        ep_size: int,
        top_k: int,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        extra_payload_bytes_per_token: int = 0,
    ) -> int:
        """Calculate workspace size per rank for MoE AlltoAll."""
        raise NotImplementedError

    def create_moe_alltoall(
        self,
        mapping: Any,
        max_num_tokens: int,
        top_k: int,
        num_slots: int,
        workspace_size_per_rank: int,
        num_experts: Optional[int] = None,
    ) -> Any:
        """Create MoE AlltoAll communication handler."""
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

    # MNNVL Operations
    def mnnvl_initialize(self):
        self._MnnvlMemory.initialize()

    def mnnvl_get_workspaces(self, mapping):
        return self._MnnvlMoe.get_moe_workspaces(mapping)

    def mnnvl_get_prepare_workspace(self, mapping):
        return self._MnnvlMoe.get_moe_prepare_workspace(mapping)

    def mnnvl_moe_alltoallv_prepare_without_allgather(
        self,
        expert_ids,
        expert_statics,
        workspace,
        max_token_count_per_rank,
        ep_rank,
        ep_size,
        expert_count,
        slot_count,
        top_k,
        scales=None,
    ):
        return self._MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
            expert_ids,
            expert_statics,
            workspace,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )

    def mnnvl_moe_alltoallv(self, x, alltoall_info, workspace, ep_rank, ep_size):
        return self._MnnvlMoe.mnnvl_moe_alltoallv(x, alltoall_info, workspace, ep_rank, ep_size)

    def mnnvl_moe_alltoallv_combine(
        self,
        x,
        alltoall_info,
        workspace,
        ep_rank,
        ep_size,
        top_k,
        token_count,
        use_low_precision_combine=False,
        do_reduce=True,
    ):
        return self._MnnvlMoe.mnnvl_moe_alltoallv_combine(
            x,
            alltoall_info,
            workspace,
            ep_rank=ep_rank,
            ep_size=ep_size,
            top_k=top_k,
            token_count=token_count,
            use_low_precision_combine=use_low_precision_combine,
            do_reduce=do_reduce,
        )

    # AlltoAll Operations
    def get_a2a_workspace_size(
        self,
        ep_size,
        top_k,
        max_num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        extra_payload_bytes_per_token: int = 0,
    ):
        return self._MoeAlltoAll.calculate_required_workspace_size(
            ep_size, top_k, max_num_tokens, hidden_size, dtype, extra_payload_bytes_per_token
        )

    def create_moe_alltoall(
        self, mapping, max_num_tokens, top_k, num_slots, workspace_size_per_rank, num_experts
    ):
        return self._MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_slots=num_slots,
            workspace_size_per_rank=workspace_size_per_rank,
            num_experts=num_experts,
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
        hidden_size = hidden_states.shape[-1]
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
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )
                if not do_finalize:
                    return outputs
                else:
                    final_hidden_states = outputs[0]
                    if final_hidden_states.shape[1] > valid_hidden_size:
                        final_hidden_states = final_hidden_states[
                            :, :valid_hidden_size
                        ].contiguous()
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
            )


# ==================== Flashinfer Backend ====================
@register_op_backend("flashinfer")
class FlashinferOpBackend(MoEOpBackend):
    """Flashinfer op backend implementation."""

    def __init__(self):
        import flashinfer.fused_moe as _flashinfer_fused_moe
        from flashinfer.comm.mnnvl import MnnvlMemory as _flashinfer_MnnvlMemory
        from flashinfer.comm.trtllm_alltoall import MnnvlMoe as _flashinfer_MnnvlMoe
        from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll as _flashinfer_MoeAlltoAll
        from flashinfer.fp4_quantization import fp4_quantize as _flashinfer_fp4_quantize
        from flashinfer.fp8_quantization import mxfp8_quantize as _flashinfer_mxfp8_quantize
        from flashinfer.fused_moe.core import ActivationType as _flashinfer_activation_type
        from flashinfer.fused_moe.core import RoutingMethodType as _flashinfer_routing_method_type

        from ..fused_moe.routing import RoutingMethodType as _trtllmgen_routing_method_type

        self._trtllmgen_routing_method_type = _trtllmgen_routing_method_type

        self._activation_type = _flashinfer_activation_type
        self._routing_method_type = _flashinfer_routing_method_type
        self._fused_moe = _flashinfer_fused_moe
        self._MnnvlMemory = _flashinfer_MnnvlMemory
        self._MnnvlMoe = _flashinfer_MnnvlMoe
        self._MoeAlltoAll = _flashinfer_MoeAlltoAll
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

    # MNNVL Operations
    def mnnvl_initialize(self):
        self._MnnvlMemory.initialize()

    def mnnvl_get_workspaces(self, mapping):
        return self._MnnvlMoe.get_moe_workspaces(mapping)

    def mnnvl_get_prepare_workspace(self, mapping):
        return self._MnnvlMoe.get_moe_prepare_workspace(mapping)

    def mnnvl_moe_alltoallv_prepare_without_allgather(
        self,
        expert_ids,
        expert_statics,
        workspace,
        max_token_count_per_rank,
        ep_rank,
        ep_size,
        expert_count,
        slot_count,
        top_k,
        scales=None,
    ):
        # flashinfer API has different signature
        alltoall_info, _, __, gathered_info = (
            self._MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                expert_ids,
                scales,
                expert_statics,
                workspace,
                max_token_count_per_rank,
                ep_rank,
                ep_size,
                expert_count,
                slot_count,
                top_k,
            )
        )
        return alltoall_info, gathered_info

    def mnnvl_moe_alltoallv(self, x, alltoall_info, workspace, ep_rank, ep_size):
        # flashinfer processes one tensor at a time
        results = []
        for tensor in x:
            if tensor is not None:
                result = self._MnnvlMoe.mnnvl_moe_alltoallv(
                    tensor, alltoall_info, workspace, ep_rank, ep_size
                )
                results.append(result)
            else:
                results.append(None)
        return results

    def mnnvl_moe_alltoallv_combine(
        self,
        x,
        alltoall_info,
        workspace,
        ep_rank,
        ep_size,
        top_k,
        token_count,
        use_low_precision_combine=False,
        do_reduce=True,
    ):
        # flashinfer doesn't support use_low_precision_combine
        assert do_reduce, "do_reduce must be True"
        assert use_low_precision_combine is False, "use_low_precision_combine must be False"
        return self._MnnvlMoe.mnnvl_moe_alltoallv_combine(
            x,
            alltoall_info,
            workspace,
            ep_rank=ep_rank,
            ep_size=ep_size,
            top_k=top_k,
            token_count=token_count,
        )

    # AlltoAll Operations
    def get_a2a_workspace_size(
        self,
        ep_size,
        top_k,
        max_num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        extra_payload_bytes_per_token: int = 0,
    ):
        workspace_size = self._MoeAlltoAll.get_moe_workspace_size_per_rank(
            ep_size, top_k, max_num_tokens, hidden_size, extra_payload_bytes_per_token
        )
        # Check for environment variable override
        workspace_mb_env = os.environ.get("TRTLLM_MOE_A2A_WORKSPACE_MB")
        if workspace_mb_env:
            workspace_size_env = int(workspace_mb_env) * 1024 * 1024
            logger.warning(
                f"Overriding automatically calculated workspace_size_per_rank ({workspace_size} bytes) with "
                f"TRTLLM_MOE_A2A_WORKSPACE_MB={workspace_mb_env} ({workspace_size_env} bytes). "
                f"Automatically calculated workspace_size_per_rank is conservatively large, "
                f"please only consider overriding it if you have a specific reason."
            )
            workspace_size = workspace_size_env
        return workspace_size

    def create_moe_alltoall(
        self, mapping, max_num_tokens, top_k, num_slots, workspace_size_per_rank, num_experts
    ):
        # flashinfer API has not updated to the new signature yet
        return self._MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_experts=num_slots,
            workspace_size_per_rank=workspace_size_per_rank,
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
        use_shuffled_weight=False,
        weight_layout=0,
        enable_pdl=None,
        tune_max_num_tokens=8192,
    ):
        assert topk_weights is not None and topk_ids is not None, (
            "topk_weights and topk_ids must be provided None"
        )
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
            routing_method_type,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            enable_pdl=enable_pdl,
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
            packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(
                torch.int16
            )
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
            return outputs
        else:
            final_hidden_states = outputs[0]
            # Slice output if it was padded
            if final_hidden_states.shape[1] > valid_hidden_size:
                final_hidden_states = final_hidden_states[:, :valid_hidden_size].contiguous()
            return final_hidden_states
