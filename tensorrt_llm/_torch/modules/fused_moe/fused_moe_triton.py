from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IS_TRITON_KERNELS_AVAILABLE = False
try:
    # WAR since the triton wheel doesn't include the triton_kernels module
    import os
    import sys
    llm_root = os.getenv('LLM_ROOT')
    if llm_root:
        # On CI, we use LLM_ROOT to locate the 3rdparty directory.
        triton_path = os.path.join(llm_root, '3rdparty', 'triton', 'python',
                                   'triton_kernels')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        triton_path = os.path.join(current_dir, '..', '..', '..', '..',
                                   '3rdparty', 'triton', 'python',
                                   'triton_kernels')
    triton_path = os.path.abspath(triton_path)
    if os.path.exists(triton_path) and triton_path not in sys.path:
        sys.path.insert(0, triton_path)
    import triton_kernels.swiglu
    from triton_kernels.matmul_ogs import (FlexCtx, MicroscalingCtx,
                                           PrecisionConfig, matmul_ogs)
    from triton_kernels.numerics import InFlexData
    from triton_kernels.numerics_details.mxfp import (
        SwizzlingType, perm_tensor_from_contig, perm_tuple_from_contig,
        swizzle_mx_scale_bw, swizzle_mxfp4_scale_hopper,
        swizzle_mxfp4_value_hopper)
    from triton_kernels.routing import routing
    IS_TRITON_KERNELS_AVAILABLE = True
except ImportError:
    pass

from ...model_config import ModelConfig
from ..linear import TensorParallelMode, load_weight_shard
from .interface import MoE
from .quantization import (FusedMoEMethodBase, MoEWeightLoadingMode,
                           load_activation_scales_fp8_qdq,
                           requantize_expert_w3_w1_weight_fp8_qdq)
from .routing import BaseMoeRoutingMethod, RenormalizeMoeRoutingMethod


def shuffle_weight_for_activation_kernel(
        w3_w1_weight: torch.Tensor) -> torch.Tensor:
    temp_weight = w3_w1_weight.clone()
    last_dim = w3_w1_weight.shape[-1]
    assert w3_w1_weight.dim() in [1, 2, 3]
    # n_dims = 1: Single expert bias (like the unquantized case)
    # n_dims = 2: Single expert weight (like the unquantized case)
    # n_dims = 3: Multiple experts weight (re-quantization for fp8 qdq)
    w3_w1_weight[..., 0::2] = temp_weight[..., last_dim // 2:]
    w3_w1_weight[..., 1::2] = temp_weight[..., 0:last_dim // 2]
    return w3_w1_weight


class TritonUnquantizedFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, shuffle_weight=True):
        super().__init__()
        self.shuffle_weight = shuffle_weight

        if not IS_TRITON_KERNELS_AVAILABLE:
            raise ImportError("Triton kernels are not available.")

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = module.dtype
        assert weight_dtype == torch.bfloat16, \
            f"TritonUnquantizedFusedMoEMethod only supports bfloat16 weights, got {weight_dtype}"

        # The Triton kernel accepts the w3_w1_weight in (num_experts, hidden_dim, intermediate_dim * 2) format
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.hidden_size,
                              module.intermediate_size_per_partition * 2)

        # The Triton kernel accepts the w2_weight in (num_experts, intermediate_dim, hidden_dim) format
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.intermediate_size_per_partition,
            module.hidden_size,
        )
        super().create_weights(module,
                               weight_dtype,
                               w3_w1_weight_shape,
                               w2_weight_shape,
                               bias_dtype=torch.float32)
        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        return tuple()

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        w1_weight_shard = load_weight_shard(w1_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_shard = load_weight_shard(w3_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)

        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)
        # We use .to here since for Triton the bias is always in float32 and a conversion is needed.
        w31_weight_shard = w31_weight_shard.to(dst_w3_w1_weight.dtype)

        # This function is shared by weights and biases, we only do transpose for weights
        if w31_weight_shard.dim() == 2:
            # Transpose the weights to match the expected format for the Triton gemm kernel
            w31_weight_shard = w31_weight_shard.transpose(0, 1).contiguous()

        if self.shuffle_weight:
            w31_weight_shard = shuffle_weight_for_activation_kernel(
                w31_weight_shard)

        dst_w3_w1_weight.copy_(w31_weight_shard, non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)
        # We use .to here since for Triton the bias is always in float32 and a conversion is needed.
        w2_weight_shard = w2_weight_shard.to(dst_w2_weight.dtype)

        # This function is shared by weights and biases, we only do transpose for weights
        if w2_weight_shard.dim() == 2:
            # Transpose the weights to match the expected format for the Triton gemm kernel
            w2_weight_shard = w2_weight_shard.transpose(0, 1).contiguous()
        else:
            assert w2_weight_shard.dim() == 1
            # Handle TP contribution of bias
            w2_weight_shard /= module.tp_size

        dst_w2_weight.copy_(w2_weight_shard, non_blocking=True)

    def apply(self, module: torch.nn.Module, x: torch.Tensor,
              router_logits: torch.Tensor) -> torch.Tensor:
        # Fetch all the data needed for the Triton kernel
        hidden_states = x
        expert_logits = router_logits
        gemm1_weights = module.w3_w1_weight
        gemm2_weights = module.w2_weight
        top_k = module.routing_method.experts_per_token

        # hidden_states: (num_tokens, hidden_dim) torch.bfloat16
        # expert_logits: (num_tokens, num_experts) torch.bfloat16
        # gemm1_weights: (num_experts, intermediate_dim * 2, hidden_dim) torch.bfloat16
        # gemm2_weights: (num_experts, hidden_dim, intermediate_dim) torch.bfloat16

        # Step 1: Routing
        num_experts = expert_logits.shape[1]
        if num_experts > 1:
            rdata, gather_indx, scatter_indx = routing(expert_logits, top_k)
        else:
            rdata, gather_indx, scatter_indx = None, None, None

        # Step 2: Gemm1
        # Setup quantization context
        pc1 = PrecisionConfig(flex_ctx=FlexCtx(),
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton kernel, which also does permutation
        gemm1_output = matmul_ogs(hidden_states,
                                  gemm1_weights,
                                  module.w3_w1_bias if module.bias else None,
                                  rdata,
                                  gather_indx=gather_indx,
                                  precision_config=pc1)

        # Step 3: Activation
        # Setup quantization context
        pcs = triton_kernels.swiglu.PrecisionConfig(limit=None)

        # Call the Triton activation kernel
        act_out = triton_kernels.swiglu.swiglu(
            gemm1_output,
            module.swiglu_alpha or 1.0,  # scale before sigmoid
            module.swiglu_beta
            or 0.0,  # bias added to the linear term of swiglu
            pcs,
            routing_data=rdata)

        # Step 4: Gemm2
        # Setup quantization context
        pc2 = PrecisionConfig(flex_ctx=FlexCtx(),
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton kernel, which also does finalization
        gemm2_output = matmul_ogs(act_out,
                                  gemm2_weights,
                                  module.w2_bias if module.bias else None,
                                  rdata,
                                  scatter_indx=scatter_indx,
                                  precision_config=pc2,
                                  gammas=rdata.gate_scal if rdata else None)
        return gemm2_output


class TritonFP8QDQFusedMoEQuantScales(NamedTuple):
    fc1_dequant: torch.Tensor
    fc2_dequant: torch.Tensor
    fc1_input_dequant: torch.Tensor
    fc2_input_dequant: torch.Tensor


# We inherit from TritonUnquantizedFusedMoEMethod to reuse the weight preprocessing logic
class TritonFP8QDQFusedMoEMethod(TritonUnquantizedFusedMoEMethod):

    def __init__(self):
        # Due to the requantization logic in the Triton kernel, we delay the shuffle
        super().__init__(shuffle_weight=False)

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = torch.float8_e4m3fn

        # The Triton kernel accepts the w3_w1_weight in (num_experts, hidden_dim, intermediate_dim * 2) format
        w3_w1_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size,
            module.intermediate_size_per_partition * 2,
        )

        # The Triton kernel accepts the w2_weight in (num_experts, intermediate_dim, hidden_dim) format
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.intermediate_size_per_partition,
            module.hidden_size,
        )
        FusedMoEMethodBase.create_weights(self,
                                          module,
                                          weight_dtype,
                                          w3_w1_weight_shape,
                                          w2_weight_shape,
                                          bias_dtype=torch.float32)

        fc31_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                    requires_grad=False)
        module.register_parameter("fc31_dequant", fc31_dequant)

        fc2_dequant = nn.Parameter(torch.empty(module.expert_size_per_partition,
                                               dtype=torch.float32),
                                   requires_grad=False)
        module.register_parameter("fc2_dequant", fc2_dequant)

        fc31_input_dequant = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)

        fc2_input_dequant = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                         requires_grad=False)
        module.register_parameter("fc2_input_dequant", fc2_input_dequant)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = TritonFP8QDQFusedMoEQuantScales(
            fc1_dequant=module.fc31_dequant,
            fc2_dequant=module.fc2_dequant,
            fc1_input_dequant=module.fc31_input_dequant,
            fc2_input_dequant=module.fc2_input_dequant,
        )

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        assert module.quant_scales is not None
        return module.quant_scales

    def load_expert_w3_w1_weight_scale_fp8_qdq(
            self, w1_weight_scale, w3_weight_scale,
            dst_w3_w1_weight_scale: torch.Tensor):
        w1_weight_scale = w1_weight_scale[...].reshape([])
        w3_weight_scale = w3_weight_scale[...].reshape([])
        dst_w3_w1_weight_scale.copy_(max(w1_weight_scale, w3_weight_scale))

    def load_expert_w2_weight_scale_fp8(self, w2_weight_scale,
                                        dst_w2_weight_scale: torch.Tensor):
        dst_w2_weight_scale.copy_(w2_weight_scale[...].reshape([]))

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        max_fc31_input_scale, max_fc2_input_scale = load_activation_scales_fp8_qdq(
            module, weights)

        # Step2: Load weight scales and requantize w3_w1_weight.
        tmp_w3_w1_weight_scale = torch.empty(module.expert_size_per_partition,
                                             dtype=torch.float32)
        tmp_w2_weight_scale = torch.empty(module.expert_size_per_partition,
                                          dtype=torch.float32)

        for local_slot_id, expert_id in enumerate(
                module.initial_local_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_weight_scale = weights[f"gate_up_proj_weight_scale"]
                w3_weight_scale = weights[f"gate_up_proj_weight_scale"]
                w2_weight_scale = weights[f"down_proj_weight_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            self.load_expert_w3_w1_weight_scale_fp8_qdq(
                w1_weight_scale, w3_weight_scale,
                tmp_w3_w1_weight_scale[expert_idx])

            # Shared code from FP8QDQFusedMoEMethod, need to pass a transposed view
            requantize_expert_w3_w1_weight_fp8_qdq(
                module, w1_weight_scale, w3_weight_scale,
                module.w3_w1_weight.data[expert_idx].transpose(0, 1))

            self.load_expert_w2_weight_scale_fp8(
                w2_weight_scale, tmp_w2_weight_scale[expert_idx])

        # now we can shuffle the weights for the activation kernel
        module.w3_w1_weight.data = shuffle_weight_for_activation_kernel(
            module.w3_w1_weight.data)
        if module.bias:
            # Bias should also be shuffled here
            module.w3_w1_bias.data = shuffle_weight_for_activation_kernel(
                module.w3_w1_bias.data)
        # Step3: calculate and store final loaded weights
        module.fc31_dequant.data.copy_(tmp_w3_w1_weight_scale)
        module.fc2_dequant.data.copy_(tmp_w2_weight_scale)
        module.fc31_input_dequant.data.copy_(max_fc31_input_scale)
        module.fc2_input_dequant.data.copy_(max_fc2_input_scale)

    def apply(self, module: torch.nn.Module, x: torch.Tensor,
              router_logits: torch.Tensor) -> torch.Tensor:
        # Fetch all the data needed for the Triton kernel
        hidden_states, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, module.fc31_input_dequant)
        hidden_states_scale = module.fc31_input_dequant
        expert_logits = router_logits
        gemm1_weights = module.w3_w1_weight
        gemm1_scales = module.fc31_dequant
        gemm2_weights = module.w2_weight
        gemm2_scales = module.fc2_dequant
        top_k = module.routing_method.experts_per_token

        # hidden_states: (num_tokens, hidden_dim) torch.float8_e4m3fn
        # hidden_states_scale: (,) torch.float32
        # expert_logits: (num_tokens, num_experts) torch.bfloat16
        # gemm1_weights: (num_experts, hidden_dim, intermediate_dim * 2) torch.float8_e4m3fn
        # gemm1_scales: (num_experts, ) torch.float32
        # gemm2_weights: (num_experts, intermediate_dim, hidden_dim) torch.float8_e4m3fn
        # gemm2_scales: (num_experts, ) torch.float32

        # Step 1: Routing
        num_experts = expert_logits.shape[1]
        if num_experts > 1:
            rdata, gather_indx, scatter_indx = routing(expert_logits, top_k)
        else:
            rdata, gather_indx, scatter_indx = None, None, None

        # Step 2: Gemm1
        # Setup quantization context
        flex_ctx_1 = FlexCtx(
            lhs_data=InFlexData(scale=hidden_states_scale),
            rhs_data=InFlexData(scale=gemm1_scales),
        )
        pc1 = PrecisionConfig(flex_ctx=flex_ctx_1,
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton kernel, which also does permutation
        gemm1_output = matmul_ogs(hidden_states,
                                  gemm1_weights,
                                  module.w3_w1_bias if module.bias else None,
                                  rdata,
                                  gather_indx=gather_indx,
                                  precision_config=pc1)

        # Step 3: Activation
        # Setup quantization context
        pcs = triton_kernels.swiglu.PrecisionConfig(limit=None)

        # Call the Triton activation kernel
        act_out = triton_kernels.swiglu.swiglu(
            gemm1_output,
            module.swiglu_alpha or 1.0,  # scale before sigmoid
            module.swiglu_beta
            or 0.0,  # bias added to the linear term of swiglu
            pcs,
            routing_data=rdata)
        # Quantize the activation output manually since the Triton activation kernel doesn't support bf16 in fp8 out
        act_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            act_out, module.fc2_input_dequant)

        # Step 4: Gemm2
        # Setup quantization context
        flex_ctx_2 = FlexCtx(
            lhs_data=InFlexData(scale=module.fc2_input_dequant),
            rhs_data=InFlexData(scale=gemm2_scales),
        )
        pc2 = PrecisionConfig(flex_ctx=flex_ctx_2,
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton kernel, which also does finalization
        gemm2_output = matmul_ogs(act_out,
                                  gemm2_weights,
                                  module.w2_bias if module.bias else None,
                                  rdata,
                                  scatter_indx=scatter_indx,
                                  precision_config=pc2,
                                  gammas=rdata.gate_scal if rdata else None)
        return gemm2_output


class TritonMXFP4FusedMoEQuantScales(NamedTuple):
    fc1_dequant: torch.Tensor
    fc2_dequant: torch.Tensor
    fc1_input_dequant: torch.Tensor
    fc2_input_dequant: torch.Tensor


def get_swizzle_type(activation_type):
    assert activation_type in [torch.float8_e4m3fn, torch.bfloat16]
    assert torch.cuda.get_device_capability()[0] >= 9
    if torch.cuda.get_device_capability()[0] < 10:
        if activation_type == torch.float8_e4m3fn:
            swizzle_mx_value = None
            swizzle_mx_scale = SwizzlingType.BLACKWELL
        else:
            swizzle_mx_value = SwizzlingType.HOPPER
            swizzle_mx_scale = SwizzlingType.HOPPER
    else:
        swizzle_mx_value = None
        swizzle_mx_scale = SwizzlingType.BLACKWELL
    return swizzle_mx_value, swizzle_mx_scale


def swizzle_weight_and_scale(weight_tensor: torch.Tensor,
                             scale_tensor: torch.Tensor,
                             swizzle_value: SwizzlingType,
                             swizzle_scale: SwizzlingType) -> torch.Tensor:
    # switch to swizzle shape
    quant_tensor = weight_tensor.transpose(1, 2).contiguous()
    scale = scale_tensor.transpose(1, 2).contiguous()
    # Swizzling
    if swizzle_value == SwizzlingType.HOPPER:
        quant_tensor = swizzle_mxfp4_value_hopper(quant_tensor,
                                                  op_idx=0,
                                                  mma_version=3)
    assert quant_tensor.is_contiguous()
    axis = 1
    swizzle_axis = 2 if swizzle_scale else None
    quant_tensor = perm_tensor_from_contig(quant_tensor, axis, swizzle_axis)
    orig_scale_shape = scale.shape
    if swizzle_scale == SwizzlingType.BLACKWELL:
        scale = swizzle_mx_scale_bw(scale, allow_pad=True)
    elif swizzle_scale == SwizzlingType.HOPPER:
        scale = swizzle_mxfp4_scale_hopper(scale, num_warps=8)
    assert scale.is_contiguous()
    scale = perm_tensor_from_contig(scale, axis, swizzle_axis)
    actual_scale_shape = perm_tuple_from_contig(orig_scale_shape, axis,
                                                swizzle_axis)
    return quant_tensor, scale, actual_scale_shape


# We inherit from TritonUnquantizedFusedMoEMethod to reuse the weight preprocessing logic
class TritonMXFP4FusedMoEMethod(TritonUnquantizedFusedMoEMethod):

    def __init__(self, activation_dtype):
        super().__init__(shuffle_weight=True)
        assert activation_dtype in [torch.float8_e4m3fn, torch.bfloat16], \
            f"TritonMXFP4FusedMoEMethod only supports float8_e4m3fn or bfloat16 activation, got {activation_dtype}"
        self.activation_dtype = activation_dtype
        self.swizzle_value, self.swizzle_scale = get_swizzle_type(
            activation_dtype)

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = torch.uint8

        # The Triton kernel accepts the w3_w1_weight in (num_experts, hidden_dim, intermediate_dim * 2) format
        w3_w1_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size // 2,  # Two mxfp4 packed to a byte
            module.intermediate_size_per_partition * 2,
        )

        w3_w1_scale_shape = (
            w3_w1_weight_shape[0],
            w3_w1_weight_shape[1] //
            16,  # block size of 32 for mxfp4, we already divided by 2 before so only divide by 16
            w3_w1_weight_shape[2],
        )

        # The Triton kernel accepts the w2_weight in (num_experts, intermediate_dim, hidden_dim) format
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.intermediate_size_per_partition //
            2,  # Two mxfp4 packed to a byte,
            module.hidden_size,
        )

        w2_scale_shape = (
            w2_weight_shape[0],
            w2_weight_shape[1] //
            16,  # block size of 32 for mxfp4, we already divided by 2 before so only divide by 16
            w2_weight_shape[2],
        )

        FusedMoEMethodBase.create_weights(self,
                                          module,
                                          weight_dtype,
                                          w3_w1_weight_shape,
                                          w2_weight_shape,
                                          bias_dtype=torch.float32)

        fc31_dequant = nn.Parameter(
            torch.empty(w3_w1_scale_shape, dtype=torch.uint8),  # mxfp8 scale
            requires_grad=False)
        module.register_parameter("fc31_dequant", fc31_dequant)

        fc2_dequant = nn.Parameter(
            torch.empty(w2_scale_shape, dtype=torch.uint8),  # mxfp8 scale
            requires_grad=False)
        module.register_parameter("fc2_dequant", fc2_dequant)

        if self.activation_dtype == torch.float8_e4m3fn:
            fc31_input_dequant = nn.Parameter(torch.tensor(1.,
                                                           dtype=torch.float32),
                                              requires_grad=False)
            module.register_parameter("fc31_input_dequant", fc31_input_dequant)

            fc2_input_dequant = nn.Parameter(torch.tensor(1.,
                                                          dtype=torch.float32),
                                             requires_grad=False)
            module.register_parameter("fc2_input_dequant", fc2_input_dequant)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = TritonMXFP4FusedMoEQuantScales(
            fc1_dequant=module.fc31_dequant,
            fc2_dequant=module.fc2_dequant,
            fc1_input_dequant=getattr(
                module, 'fc31_input_dequant',
                None),  # activation scale exists only for float8_e4m3fn
            fc2_input_dequant=getattr(
                module, 'fc2_input_dequant',
                None),  # activation scale exists only for float8_e4m3fn
        )

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        assert module.quant_scales is not None
        return module.quant_scales

    def load_expert_w3_w1_weight_scale_mxfp4(
            self, w1_weight_scale, w3_weight_scale,
            dst_w3_w1_weight_scale: torch.Tensor):
        # (intermediate_dim * 2, hidden_dim / 32)
        combined_scale = torch.cat([w3_weight_scale, w1_weight_scale], dim=0)
        # (hidden_dim / 32, intermediate_dim * 2)
        combined_scale = combined_scale.transpose(0, 1)
        dst_w3_w1_weight_scale.copy_(combined_scale)

    def load_expert_w2_weight_scale_mxfp4(self, w2_weight_scale,
                                          dst_w2_weight_scale: torch.Tensor):
        w2_weight_scale = w2_weight_scale.transpose(
            0, 1)  # (intermediate_dim / 32, hidden_dim)
        dst_w2_weight_scale.copy_(w2_weight_scale)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        if self.activation_dtype == torch.float8_e4m3fn:
            try:
                max_fc31_input_scale, max_fc2_input_scale = load_activation_scales_fp8_qdq(
                    module, weights)
            except KeyError:
                # We will use dynamic quantization
                max_fc31_input_scale = None
                max_fc2_input_scale = None

        # Step2: Load weight scales
        tmp_w3_w1_weight_scale = torch.empty(module.fc31_dequant.shape,
                                             dtype=torch.uint8)
        tmp_w2_weight_scale = torch.empty(module.fc2_dequant.shape,
                                          dtype=torch.uint8)

        for local_slot_id, expert_id in enumerate(
                module.initial_local_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_weight_scale = weights[f"gate_up_proj_weight_scale"]
                w3_weight_scale = weights[f"gate_up_proj_weight_scale"]
                w2_weight_scale = weights[f"down_proj_weight_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            self.load_expert_w3_w1_weight_scale_mxfp4(
                w1_weight_scale, w3_weight_scale,
                tmp_w3_w1_weight_scale[expert_idx])

            self.load_expert_w2_weight_scale_mxfp4(
                w2_weight_scale, tmp_w2_weight_scale[expert_idx])

        # Scales need to be shuffled as well
        tmp_w3_w1_weight_scale = shuffle_weight_for_activation_kernel(
            tmp_w3_w1_weight_scale)

        # For Hopper style swizzle, we need to pad the out dim to multiple of 256
        def _maybe_pad_weight_and_scale(weight, scale):
            if self.swizzle_scale != SwizzlingType.HOPPER:
                return weight, scale
            out_dim = weight.shape[-1]
            assert scale.shape[
                -1] == out_dim, "Out dim of weight and scale should match"
            pad_size = (256 - out_dim % 256) % 256
            weight = F.pad(
                weight,
                (0,
                 pad_size)).contiguous()  # Pad the last dimension on right side
            scale = F.pad(scale, (0, pad_size)).contiguous()
            return weight, scale

        tmp_w3_w1_weight, tmp_w3_w1_weight_scale = _maybe_pad_weight_and_scale(
            module.w3_w1_weight, tmp_w3_w1_weight_scale)

        tmp_w2_weight, tmp_w2_weight_scale = _maybe_pad_weight_and_scale(
            module.w2_weight, tmp_w2_weight_scale)

        # Apply swizzle to the scales
        tmp_w3_w1_weight, tmp_w3_w1_weight_scale, tmp_w3_w1_scale_shape = swizzle_weight_and_scale(
            tmp_w3_w1_weight, tmp_w3_w1_weight_scale, self.swizzle_value,
            self.swizzle_scale)
        tmp_w2_weight, tmp_w2_weight_scale, tmp_w2_scale_shape = swizzle_weight_and_scale(
            tmp_w2_weight, tmp_w2_weight_scale, self.swizzle_value,
            self.swizzle_scale)

        # Step3: store final loaded weights and scales
        # Don't use copy_ here, it will break the swizzle stride
        module.w3_w1_weight.data = tmp_w3_w1_weight
        device = module.w3_w1_weight.device
        module.fc31_dequant.data = tmp_w3_w1_weight_scale.to(device)
        self.w3_w1_scale_shape = tmp_w3_w1_scale_shape  # Triton swizzle needs this original shape as the swizzle may not keep the original shape
        module.w2_weight.data = tmp_w2_weight
        module.fc2_dequant.data = tmp_w2_weight_scale.to(device)
        self.w2_scale_shape = tmp_w2_scale_shape
        if self.activation_dtype == torch.float8_e4m3fn:
            if max_fc31_input_scale is None or max_fc2_input_scale is None:
                module.fc31_input_dequant = None
                module.fc2_input_dequant = None
            else:
                module.fc31_input_dequant.data.copy_(max_fc31_input_scale)
                module.fc2_input_dequant.data.copy_(max_fc2_input_scale)

    def apply(self, module: torch.nn.Module, x: torch.Tensor,
              router_logits: torch.Tensor) -> torch.Tensor:
        # Fetch all the data needed for the Triton kernel
        if self.activation_dtype == torch.float8_e4m3fn:
            if module.fc31_input_dequant is None:
                hidden_states, hidden_states_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                    x)
            else:
                hidden_states, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, module.fc31_input_dequant)
                hidden_states_scale = module.fc31_input_dequant
        else:
            hidden_states = x
        expert_logits = router_logits
        gemm1_weights = module.w3_w1_weight
        gemm1_scales = module.fc31_dequant
        gemm2_weights = module.w2_weight
        gemm2_scales = module.fc2_dequant
        top_k = module.routing_method.experts_per_token

        # hidden_states: (num_tokens, hidden_dim) torch.float8_e4m3fn
        # hidden_states_scale: (,) torch.float32
        # expert_logits: (num_tokens, num_experts) torch.bfloat16
        # gemm1_weights: (num_experts, hidden_dim / 2, intermediate_dim * 2) torch.uint8
        # gemm1_scales: (num_experts, hidden_dim / 32, intermediate_dim * 2) torch.uint8
        # gemm2_weights: (num_experts, intermediate_dim / 2, hidden_dim) torch.uint8
        # gemm2_scales: (num_experts, intermediate_dim / 32, hidden_dim) torch.float32

        # Step 1: Routing
        num_experts = expert_logits.shape[1]
        if num_experts > 1:
            rdata, gather_indx, scatter_indx = routing(expert_logits, top_k)
        else:
            rdata, gather_indx, scatter_indx = None, None, None

        # Step 2: Gemm1
        # Setup quantization context
        mx_ctx_1 = MicroscalingCtx(
            weight_scale=gemm1_scales,
            swizzle_value=self.swizzle_value,
            swizzle_scale=self.swizzle_scale,
            actual_weight_scale_shape=self.w3_w1_scale_shape)
        if self.activation_dtype == torch.float8_e4m3fn:
            flex_ctx_1 = FlexCtx(
                lhs_data=InFlexData(scale=hidden_states_scale), )
        else:
            flex_ctx_1 = FlexCtx()
        pc1 = PrecisionConfig(mx_ctx=mx_ctx_1,
                              flex_ctx=flex_ctx_1,
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton kernel, which also does permutation
        gemm1_output = matmul_ogs(hidden_states,
                                  gemm1_weights,
                                  module.w3_w1_bias if module.bias else None,
                                  rdata,
                                  gather_indx=gather_indx,
                                  precision_config=pc1)

        def _maybe_remove_padding(gemm_output, expected_size):
            assert gemm_output.dim() == 2
            if gemm_output.shape[-1] != expected_size:
                assert self.swizzle_scale == SwizzlingType.HOPPER, "Only Hopper style swizzle can have padding"
                assert gemm_output.shape[
                    -1] % 256 == 0, "The padding is not done correctly"
                gemm_output = gemm_output[:, :expected_size]
            return gemm_output

        gemm1_output = _maybe_remove_padding(
            gemm1_output,
            module.intermediate_size_per_partition * 2).contiguous()

        # Step 3: Activation
        # Setup quantization context
        pcs = triton_kernels.swiglu.PrecisionConfig(limit=None)

        # Call the Triton activation kernel
        act_out = triton_kernels.swiglu.swiglu(
            gemm1_output,
            module.swiglu_alpha or 1.0,  # scale before sigmoid
            module.swiglu_beta
            or 0.0,  # bias added to the linear term of swiglu
            pcs,
            routing_data=rdata)

        if self.activation_dtype == torch.float8_e4m3fn:
            # Quantize the activation output manually since the Triton activation kernel doesn't support bf16 in fp8 out
            if module.fc2_input_dequant is None:
                act_out, act_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                    act_out)
            else:
                act_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    act_out, module.fc2_input_dequant)
                act_scale = module.fc2_input_dequant

        # Step 4: Gemm2
        # Setup quantization context
        mx_ctx_2 = MicroscalingCtx(
            weight_scale=gemm2_scales,
            swizzle_value=self.swizzle_value,
            swizzle_scale=self.swizzle_scale,
            actual_weight_scale_shape=self.w2_scale_shape)
        if self.activation_dtype == torch.float8_e4m3fn:
            flex_ctx_2 = FlexCtx(lhs_data=InFlexData(scale=act_scale), )
        else:
            flex_ctx_2 = FlexCtx()
        pc2 = PrecisionConfig(mx_ctx=mx_ctx_2,
                              flex_ctx=flex_ctx_2,
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton kernel, which also does finalization
        gemm2_output = matmul_ogs(act_out,
                                  gemm2_weights,
                                  module.w2_bias if module.bias else None,
                                  rdata,
                                  scatter_indx=scatter_indx,
                                  precision_config=pc2,
                                  gammas=rdata.gate_scal if rdata else None)

        gemm2_output = _maybe_remove_padding(gemm2_output, module.hidden_size)
        return gemm2_output


class TritonFusedMoE(MoE):

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
        aux_stream: Optional[torch.cuda.Stream] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        override_quant_method=None,
    ):
        # Override the quantization method if needed for test purpose
        # TODO(dongfengy): Remove this when we have all the quantization classes and enums for mxfp4
        if override_quant_method is not None:
            self.override_quant_method = override_quant_method

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
        )

        assert isinstance(self.routing_method, RenormalizeMoeRoutingMethod), \
            "routing_method must be an instance of RenormalizeMoeRoutingMethod for TritonFusedMoE"
        assert not self.smart_router, "Smart router is not supported in TritonFusedMoE."
        assert not self.use_dp, "AttentionDP is not supported in TritonFusedMoE."
        assert self.ep_size == 1, " TritonFusedMoE does not support expert parallelism (ep_size > 1)."

        self.num_slots = self.num_experts
        self.expert_size_per_partition = self.num_experts // self.ep_size
        self.initial_global_assignments = [
            (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
            self.num_experts for ep_rank in range(self.ep_size)
            for local_slot_id in range(self.expert_size_per_partition)
        ]
        self.slot_start = self.ep_rank * self.expert_size_per_partition
        self.slot_end = self.slot_start + self.expert_size_per_partition
        self.initial_local_expert_ids = self.initial_global_assignments[
            self.slot_start:self.slot_end]
        assert len(
            self.initial_local_expert_ids) == self.expert_size_per_partition

        self.bias = bias

        def _maybe_squeeze_act_param(p):
            if p is None or isinstance(p, (int, float)):
                return p
            assert isinstance(p, torch.Tensor)
            assert p.dtype == torch.float32
            assert p.shape == (self.num_experts, 1)
            assert torch.all(
                p == p[0]
            ), "All experts must have the same swiglu alpha/beta for Triton kernel"
            p = p[0].item()
            return p

        self.swiglu_alpha = _maybe_squeeze_act_param(swiglu_alpha)
        self.swiglu_beta = _maybe_squeeze_act_param(swiglu_beta)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _get_quant_method(self):
        if hasattr(self, 'override_quant_method'):
            return self.override_quant_method()

        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_qdq():
                return TritonFP8QDQFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                return TritonMXFP4FusedMoEMethod(
                    activation_dtype=torch.float8_e4m3fn)
            else:
                raise ValueError(
                    f"Unsupported quantization mode: {self.quant_config.quant_mode}"
                )
        else:
            return TritonUnquantizedFusedMoEMethod()

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # TODO(dongfengy): Add missing comm primitives for TP/EP
        hidden_states = self.quant_method.apply(self, x, router_logits)

        if self.tp_size > 1:
            assert self.reduce_results
            hidden_states = hidden_states.contiguous(
            )  # There might be padding going on
            hidden_states = self.all_reduce(hidden_states)

        return hidden_states

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)
