from __future__ import annotations

import os
import sys
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

IS_TRITON_KERNELS_AVAILABLE = False
# We expect to find triton_kernels under $TRITON_ROOT/python/triton_kernels
# Triton upstream commit f3067cd3bd0c29065fa4ecdb724b6f29cbabea5f has been verified.
triton_root = os.getenv('TRITON_ROOT')
if triton_root:
    triton_root = os.path.abspath(
        os.path.join(triton_root, 'python', 'triton_kernels'))
    if os.path.exists(triton_root) and triton_root not in sys.path:
        sys.path.insert(0, triton_root)
    assert triton.__version__ >= "3.4.0", "Triton kernels are detected but the Triton wheel is too old"
    import triton_kernels.swiglu
    from triton_kernels.matmul_ogs import (FlexCtx, FnSpecs, FusedActivation,
                                           PrecisionConfig, matmul_ogs)
    from triton_kernels.numerics import InFlexData
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    IS_TRITON_KERNELS_AVAILABLE = True

from ...model_config import ModelConfig
from ..linear import TensorParallelMode, load_weight_shard
from .interface import MoE
from .quantization import (FusedMoEMethodBase, MoEWeightLoadingMode,
                           load_activation_scales_fp8_qdq,
                           requantize_expert_w3_w1_weight_fp8_qdq)
from .routing import BaseMoeRoutingMethod, RenormalizeMoeRoutingMethod


# Triton kernels has hardcoded beta = 1, so we use this implementation when beta is not 1
def swiglu_torch(a: torch.Tensor, alpha: float, beta: float,
                 limit: Optional[float]) -> torch.Tensor:
    a_glu = a[..., ::2]
    if limit is not None:
        a_glu = a_glu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_glu = a_glu * torch.sigmoid(alpha * a_glu)
    out = out_glu * (a_linear + beta)
    return out


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


# This kernel remaps the global routing information (bitmatrix and indices)
# to a local view for this specific EP worker.
#
# The bitmask is shifted so that the worker's slice of experts starts at bit 0.
# Since the slice may not align with 32-bit word boundaries, this is done by
# loading two consecutive words (v1, v2) and "stitching" the result together.
# The expression `(v1 >> start_bit) | (v2 << (32 - start_bit))` takes the
# upper bits from v1 and combines them with the lower bits from v2 to form the
# new, correctly aligned word.
@triton.jit
def _routing_shift_bitmatrix_range(Bitmatrix, stride_bm, stride_bn, Indices,
                                   stride_im, stride_in, n_words, n_cols,
                                   slice_start, slice_end,
                                   BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    start_word = slice_start // 32
    start_bit = slice_start % 32

    for col0 in range(0, n_words, BLOCK_N):
        w = col0 + tl.arange(0, BLOCK_N)  # dst‐word indices
        dst_mask = w < n_words

        # corresponding source words (and the next word for carry bits)
        src1_w = start_word + w
        src2_w = src1_w + 1

        ptr1 = Bitmatrix + pid_m * stride_bm + src1_w * stride_bn
        ptr2 = Bitmatrix + pid_m * stride_bm + src2_w * stride_bn

        v1 = tl.load(ptr1, mask=src1_w < n_words, other=0).to(tl.uint32)
        v2 = tl.load(ptr2, mask=src2_w < n_words, other=0).to(tl.uint32)

        # shift the slice down to bit‐0
        shifted = tl.where(start_bit == 0, v1,
                           (v1 >> start_bit) | (v2 << (32 - start_bit)))

        # write back in place; bits past the region are already zero
        tl.store(Bitmatrix + pid_m * stride_bm + w * stride_bn,
                 shifted.to(tl.int32),
                 mask=dst_mask)

    # Fix the indices associated with the bitmatrix.
    for col0 in range(0, n_cols, BLOCK_N):
        offs = col0 + tl.arange(0, BLOCK_N)
        mask_i = offs < n_cols

        ptr = Indices + pid_m * stride_im + offs * stride_in
        yi = tl.load(ptr, mask=mask_i, other=0).to(tl.int32)

        yi = tl.where(yi < slice_end, yi - slice_start,
                      yi)  # shift inside slice
        yi = tl.where(yi < 0, yi + slice_end, yi)  # wrap negatives

        tl.store(ptr, yi, mask=mask_i)


class TritonEPRouter():

    def prune_routing_ep(self, expt_scal, expt_indx, bitmatrix, n_expts_tot,
                         slice_start, slice_end):
        from triton_kernels.compaction import compaction
        from triton_kernels.routing import _routing_clear_bitmatrix
        n_tokens_pad = expt_scal.shape[0]
        _routing_shift_bitmatrix_range[(n_tokens_pad, )](
            bitmatrix.storage.data,
            bitmatrix.storage.data.stride(0),
            bitmatrix.storage.data.stride(1),
            expt_indx,
            expt_indx.stride(0),
            expt_indx.stride(1),
            bitmatrix.storage.data.shape[1],
            expt_indx.shape[1],
            slice_start,
            slice_end,
            BLOCK_N=512,
        )
        _routing_clear_bitmatrix[(n_tokens_pad, )](
            bitmatrix.storage.data,
            bitmatrix.storage.data.stride(0),
            bitmatrix.storage.data.stride(1),
            bitmatrix.storage.data.shape[1],
            slice_end - slice_start,
            BLOCK_N=512,
        )
        # perform compaction to update expt_scal / expt_indx
        expt_scal, expt_indx = compaction(expt_scal, expt_indx, bitmatrix)
        n_expts_tot = slice_end - slice_start
        bitmatrix.shape[-1] = n_expts_tot
        return expt_scal, expt_indx, bitmatrix

    def __call__(self,
                 logits,
                 n_expts_act,
                 sm_first=False,
                 expt_indx=None,
                 ep=1,
                 node_idx=0,
                 n_rows=None):
        n_expts_tot = logits.shape[-1]
        n_expts_local = n_expts_tot // ep
        slice_start = node_idx * n_expts_local
        slice_end = slice_start + n_expts_local

        from triton_kernels.routing import routing_from_bitmatrix
        from triton_kernels.topk import topk
        if sm_first:
            logits = torch.softmax(logits, dim=-1)
        expt_scal, expt_indx, bitmatrix = topk(logits,
                                               n_expts_act,
                                               apply_softmax=not sm_first,
                                               y_indx=expt_indx,
                                               n_rows=n_rows)
        # mutate bitmatrix
        if ep > 1:
            expt_scal, expt_indx, bitmatrix = self.prune_routing_ep(
                expt_scal, expt_indx, bitmatrix, n_expts_tot, slice_start,
                slice_end)
        return routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx,
                                      n_expts_local, n_expts_act)


def maybe_update_stride(weight):
    assert weight.dim() == 3
    # For the latest Triton kernels, w.stride(-2)==1 works universally
    return weight.transpose(1, 2).contiguous().transpose(1, 2)


class TritonUnquantizedFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, shuffle_weight=True):
        super().__init__()
        self.shuffle_weight = shuffle_weight

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

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w3_w1_weight.device
        assert device.type == "cuda"

        w1_weight_shard = load_weight_shard(w1_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_shard = load_weight_shard(w3_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

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
        device = dst_w2_weight.device
        assert device.type == "cuda"
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
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

    def load_expert_weights_to_dst(
            self, module: torch.nn.Module, weights: List[Dict],
            weight_loading_mode: MoEWeightLoadingMode,
            load_expert_ids: List[int], dst_w3_w1_weights_tensor: torch.Tensor,
            dst_w2_weights_tensor: torch.Tensor,
            dst_w3_w1_bias_tensor: Optional[torch.Tensor],
            dst_w2_bias_tensor: Optional[torch.Tensor]):
        FusedMoEMethodBase.load_expert_weights_to_dst(
            self, module, weights, weight_loading_mode, load_expert_ids,
            dst_w3_w1_weights_tensor, dst_w2_weights_tensor,
            dst_w3_w1_bias_tensor, dst_w2_bias_tensor)
        module.w3_w1_weight.data = maybe_update_stride(module.w3_w1_weight.data)
        module.w2_weight.data = maybe_update_stride(module.w2_weight.data)

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
            rdata, gather_indx, scatter_indx = TritonEPRouter()(
                expert_logits,
                top_k,
                ep=module.ep_size,
                node_idx=module.ep_rank)
        else:
            rdata, gather_indx, scatter_indx = None, None, None

        # Step 2: Gemm1
        # Setup quantization context
        pc1 = PrecisionConfig(flex_ctx=FlexCtx(),
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton gemm kernel, which also does permutation and activation
        alpha = module.swiglu_alpha or 1.0
        beta = module.swiglu_beta or 0.0
        if beta == 1.0:
            act = FusedActivation(
                FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn,
                        ("alpha", "limit")), (alpha, module.swiglu_limit), 2)
            act_out = matmul_ogs(hidden_states,
                                 gemm1_weights,
                                 module.w3_w1_bias if module.bias else None,
                                 rdata,
                                 gather_indx=gather_indx,
                                 precision_config=pc1,
                                 fused_activation=act)
        else:
            act_out = matmul_ogs(hidden_states,
                                 gemm1_weights,
                                 module.w3_w1_bias if module.bias else None,
                                 rdata,
                                 gather_indx=gather_indx,
                                 precision_config=pc1)
            act_out = swiglu_torch(act_out, alpha, beta, module.swiglu_limit)

        # Step 3: Gemm2
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

    def load_expert_w3_w1_weight_scale_fp8_qdq(
            self, w1_weight_scale, w3_weight_scale,
            dst_w3_w1_weight_scale: torch.Tensor):
        w1_weight_scale = w1_weight_scale[...].reshape([])
        w3_weight_scale = w3_weight_scale[...].reshape([])
        dst_w3_w1_weight_scale.copy_(max(w1_weight_scale, w3_weight_scale),
                                     non_blocking=True)

    def load_expert_w2_weight_scale_fp8(self, w2_weight_scale,
                                        dst_w2_weight_scale: torch.Tensor):
        dst_w2_weight_scale.copy_(w2_weight_scale[...].reshape([]),
                                  non_blocking=True)

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
        module.fc31_dequant.data.copy_(tmp_w3_w1_weight_scale,
                                       non_blocking=True)
        module.fc2_dequant.data.copy_(tmp_w2_weight_scale, non_blocking=True)
        module.fc31_input_dequant.data.copy_(max_fc31_input_scale,
                                             non_blocking=True)
        module.fc2_input_dequant.data.copy_(max_fc2_input_scale,
                                            non_blocking=True)

    def load_expert_weights_to_dst(
            self, module: torch.nn.Module, weights: List[Dict],
            weight_loading_mode: MoEWeightLoadingMode,
            load_expert_ids: List[int], dst_w3_w1_weights_tensor: torch.Tensor,
            dst_w2_weights_tensor: torch.Tensor,
            dst_w3_w1_bias_tensor: Optional[torch.Tensor],
            dst_w2_bias_tensor: Optional[torch.Tensor]):
        FusedMoEMethodBase.load_expert_weights_to_dst(
            self, module, weights, weight_loading_mode, load_expert_ids,
            dst_w3_w1_weights_tensor, dst_w2_weights_tensor,
            dst_w3_w1_bias_tensor, dst_w2_bias_tensor)
        module.w3_w1_weight.data = maybe_update_stride(module.w3_w1_weight.data)
        module.w2_weight.data = maybe_update_stride(module.w2_weight.data)

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
            rdata, gather_indx, scatter_indx = TritonEPRouter()(
                expert_logits,
                top_k,
                ep=module.ep_size,
                node_idx=module.ep_rank)
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

        # Call the Triton gemm kernel, which also does permutation and activation
        alpha = module.swiglu_alpha or 1.0
        beta = module.swiglu_beta or 0.0
        if beta == 1.0:
            act = FusedActivation(
                FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn,
                        ("alpha", "limit")), (alpha, module.swiglu_limit), 2)
            act_out = matmul_ogs(hidden_states,
                                 gemm1_weights,
                                 module.w3_w1_bias if module.bias else None,
                                 rdata,
                                 gather_indx=gather_indx,
                                 precision_config=pc1,
                                 fused_activation=act)
        else:
            act_out = matmul_ogs(hidden_states,
                                 gemm1_weights,
                                 module.w3_w1_bias if module.bias else None,
                                 rdata,
                                 gather_indx=gather_indx,
                                 precision_config=pc1)
            act_out = swiglu_torch(act_out, alpha, beta, module.swiglu_limit)

        # Quantize the activation output manually since the Triton activation kernel doesn't support bf16 in fp8 out
        act_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            act_out, module.fc2_input_dequant)

        # Step 3: Gemm2
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


def swizzle_weight_and_scale(w: torch.Tensor, w_scale: torch.Tensor):
    # (num_experts, in_dim//2, out_dim)
    w_shape = w.shape
    # (num_experts, in_dim//32, out_dim)
    w_scale_shape = w_scale.shape
    assert w_shape[0] == w_scale_shape[0]
    assert w_shape[1] * 2 == w_scale_shape[1] * 32
    assert w_shape[2] == w_scale_shape[2]
    w = maybe_update_stride(w)
    #num_warps = 4 if batch <= 512 else 8
    num_warps = int(os.getenv("TRITON_MOE_MXFP4_NUM_WARPS", 4))
    assert num_warps in [4, 8], \
        f"TRITON_MOE_MXFP4_NUM_WARPS should be 4 or 8, got {num_warps}"
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1)
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps)
    # swizzling path is broken for H20
    if torch.cuda.get_device_name() == "NVIDIA H20":
        from triton_kernels.tensor_details.layout_details.strided import \
            StridedLayout
        value_layout = StridedLayout
        value_layout_opts = dict()
        scale_layout = StridedLayout
        scale_layout_opts = dict()

    opt = {"value_layout": value_layout, "value_layout_opts": value_layout_opts, \
            "scale_layout": scale_layout, "scale_layout_opts": scale_layout_opts}

    # w, w_scale = downcast_to_mxfp(tensor.to(torch.bfloat16), torch.uint8, axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), opt["value_layout"],
                       **opt["value_layout_opts"])
    w_scale = convert_layout(wrap_torch_tensor(w_scale), opt["scale_layout"],
                             **opt["scale_layout_opts"])
    return w, w_scale


def get_padded_size(size: int, padding: int) -> int:
    return ((size + padding - 1) // padding) * padding


# Pad both n and k dimensions, then shard along shard_axis
# Handles weights, scales, and biases, which are expected to be 1D or 2D tensors.
def shard_and_pad_tensor(
    tensor: torch.Tensor,
    shard_axis: int,
    n_alignment: int,
    k_alignment: int,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    assert tensor.dim() in (1,
                            2), "Expecting single expect gemm weights or biases"
    assert shard_axis in (0, 1), "Shard axis must be 0 or 1"

    padding = [n_alignment, k_alignment]
    size_to_pad = [0] * 2

    tensor = tensor.to(device)

    # First we pad the sharded axis
    if shard_axis < tensor.dim():
        padded_size = get_padded_size(tensor.shape[shard_axis],
                                      padding[shard_axis] * tp_size)
        assert 0 <= tp_rank < tp_size
        assert tensor.shape[shard_axis] <= padded_size
        assert padded_size % tp_size == 0

        shard_size = padded_size // tp_size
        shard_start = tp_rank * shard_size
        assert shard_start < tensor.shape[shard_axis]
        shard_end = min(shard_start + shard_size, tensor.shape[shard_axis])
        actual_size = shard_end - shard_start

        tensor = tensor.narrow(shard_axis, shard_start, actual_size)
        size_to_pad[shard_axis] = shard_size - actual_size

    # Now we pad the non-sharded axis
    non_shard_axis = 1 - shard_axis
    if non_shard_axis < tensor.dim():
        padded_size = get_padded_size(tensor.shape[non_shard_axis],
                                      padding[non_shard_axis])
        size_to_pad[non_shard_axis] = padded_size - tensor.shape[non_shard_axis]

    # Actually call pad
    if any(size_to_pad):
        pad = (0, size_to_pad[0]) if tensor.dim() == 1 else (0, size_to_pad[1],
                                                             0, size_to_pad[0])
        tensor = torch.nn.functional.pad(tensor, pad)

    return tensor


# We inherit from TritonUnquantizedFusedMoEMethod to reuse the weight preprocessing logic
class TritonMXFP4FusedMoEMethod(TritonUnquantizedFusedMoEMethod):

    def __init__(self, activation_dtype):
        super().__init__(shuffle_weight=True)
        assert activation_dtype in [torch.float8_e4m3fn, torch.bfloat16], \
            f"TritonMXFP4FusedMoEMethod only supports float8_e4m3fn or bfloat16 activation, got {activation_dtype}"
        self.activation_dtype = activation_dtype

        self.k_alignment = 128
        self.n_alignment = 2 * self.k_alignment

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = torch.uint8

        # The Triton kernel accepts the w3_w1_weight in (num_experts, hidden_dim, intermediate_dim * 2) format
        w3_w1_weight_shape = (
            module.expert_size_per_partition,
            get_padded_size(module.hidden_size, self.k_alignment) //
            2,  # Two mxfp4 packed to a byte
            get_padded_size(module.intermediate_size_per_partition * 2,
                            self.n_alignment),
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
            get_padded_size(module.intermediate_size_per_partition,
                            self.k_alignment) //
            2,  # Two mxfp4 packed to a byte,
            get_padded_size(module.hidden_size, self.n_alignment),
        )

        w2_scale_shape = (
            w2_weight_shape[0],
            w2_weight_shape[1] //
            16,  # block size of 32 for mxfp4, we already divided by 2 before so only divide by 16
            w2_weight_shape[2],
        )

        w3_w1_bias_shape = (w3_w1_weight_shape[0], w3_w1_weight_shape[2])
        w2_bias_shape = (w2_weight_shape[0], w2_weight_shape[2])

        FusedMoEMethodBase.create_weights(self,
                                          module,
                                          weight_dtype,
                                          w3_w1_weight_shape,
                                          w2_weight_shape,
                                          bias_dtype=torch.float32,
                                          w3_w1_bias_shape=w3_w1_bias_shape,
                                          w2_bias_shape=w2_bias_shape)

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

    def load_expert_weights_to_dst(
            self, module: torch.nn.Module, weights: List[Dict],
            weight_loading_mode: MoEWeightLoadingMode,
            load_expert_ids: List[int], dst_w3_w1_weights_tensor: torch.Tensor,
            dst_w2_weights_tensor: torch.Tensor,
            dst_w3_w1_bias_tensor: Optional[torch.Tensor],
            dst_w2_bias_tensor: Optional[torch.Tensor]):
        # dynamic quant scales for weights
        self.w3_scales = {}
        self.w1_scales = {}
        self.w2_scales = {}
        # Multithread weight load is superseded by prefetch_files() in model_engine.py
        # Also, threading adds overhead in order to protect shuffle index cache with critical section.
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            # expert_idx is the local slot index of current rank
            expert_idx = local_slot_id

            if weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight = weights[f"{expert_id}.w1.weight"]
                w3_weight = weights[f"{expert_id}.w3.weight"]
                w2_weight = weights[f"{expert_id}.w2.weight"]
                if module.bias:
                    w1_bias = weights[f"{expert_id}.w1.bias"]
                    w3_bias = weights[f"{expert_id}.w3.bias"]
                    w2_bias = weights[f"{expert_id}.w2.bias"]
            elif weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_w3_weight = weights["gate_up_proj"][expert_id].transpose(
                    0, 1)
                w1_weight, w3_weight = w1_w3_weight.chunk(2, dim=0)
                w2_weight = weights["down_proj"][expert_id].transpose(
                    0, 1).contiguous()
                if module.bias:
                    w1_w3_bias = weights["gate_up_proj.bias"][expert_id]
                    w1_bias, w3_bias = w1_w3_bias.chunk(2, dim=0)
                    w2_bias = weights["down_proj.bias"][expert_id]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {weight_loading_mode}"
                )

            w3_scale, w1_scale = self.load_expert_w3_w1_weight(
                module, w1_weight, w3_weight,
                dst_w3_w1_weights_tensor[expert_idx])

            w2_scale = self.load_expert_w2_weight(
                module, w2_weight, dst_w2_weights_tensor[expert_idx])
            if w3_scale is not None:
                self.w3_scales[expert_id] = w3_scale
            if w1_scale is not None:
                self.w1_scales[expert_id] = w1_scale
            if w2_scale is not None:
                self.w2_scales[expert_id] = w2_scale

            if module.bias:
                self.load_expert_w3_w1_weight(
                    module,
                    w1_bias,
                    w3_bias,
                    dst_w3_w1_bias_tensor.data[expert_idx],
                    is_bias=True)

                self.load_expert_w2_weight(module,
                                           w2_bias,
                                           dst_w2_bias_tensor.data[expert_idx],
                                           is_bias=True)

    def _permute_mxfp4_quantize(self, tensor):
        tensor = tensor.transpose(-2, -1).contiguous()
        tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor,
                                                           torch.uint8,
                                                           axis=-2)
        return tensor_fp4, tensor_scales

    def load_expert_w3_w1_weight(self,
                                 module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor,
                                 is_bias: bool = False):
        """
        Load w1 and w3 weights for each expert.
        """
        device = dst_w3_w1_weight.device
        assert device.type == "cuda"
        # Use full k-padding for float tensors, half for already-packed uint8
        k_pad = self.k_alignment // 2 if w1_weight.dtype == torch.uint8 else self.k_alignment
        # n is halved per-branch because we concatenate w1/w3 along N later
        n_pad = self.n_alignment // 2
        w1_weight_shard = shard_and_pad_tensor(w1_weight,
                                               0,
                                               n_pad,
                                               k_pad,
                                               module.tp_size,
                                               module.tp_rank,
                                               device=device)
        w3_weight_shard = shard_and_pad_tensor(w3_weight,
                                               0,
                                               n_pad,
                                               k_pad,
                                               module.tp_size,
                                               module.tp_rank,
                                               device=device)

        if not is_bias and w3_weight_shard.dtype in (torch.bfloat16,
                                                     torch.float16,
                                                     torch.float32):
            # [N, K] -> [K, N]
            w3_weight_shard, w3_scales = self._permute_mxfp4_quantize(
                w3_weight_shard)
            w1_weight_shard, w1_scales = self._permute_mxfp4_quantize(
                w1_weight_shard)
            cat_dim = 1
        else:
            # [N, K]
            w3_scales = None
            w1_scales = None
            cat_dim = 0

        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard],
                                     dim=cat_dim)

        # This function is shared by weights and biases, we only do transpose for weights
        if not is_bias and cat_dim == 0:
            # Transpose the weights to match the expected format for the Triton gemm kernel
            w31_weight_shard = w31_weight_shard.transpose(0, 1).contiguous()
        else:
            # We use .to here since for Triton the bias is always in float32 and a conversion is needed.
            w31_weight_shard = w31_weight_shard.to(dst_w3_w1_weight.dtype)

        if self.shuffle_weight:
            w31_weight_shard = shuffle_weight_for_activation_kernel(
                w31_weight_shard)

        dst_w3_w1_weight.copy_(w31_weight_shard, non_blocking=True)
        return (w3_scales, w1_scales)

    def load_expert_w2_weight(self,
                              module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor,
                              is_bias: bool = False):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w2_weight.device
        assert device.type == "cuda"
        k_pad = self.k_alignment // 2 if w2_weight.dtype == torch.uint8 else self.k_alignment
        w2_weight_shard = shard_and_pad_tensor(w2_weight,
                                               1,
                                               self.n_alignment,
                                               k_pad,
                                               module.tp_size,
                                               module.tp_rank,
                                               device=device)
        w2_scales = None

        if is_bias:
            # We use .to here since for Triton the bias is always in float32 and a conversion is needed.
            w2_weight_shard = w2_weight_shard.to(dst_w2_weight.dtype)
            assert w2_weight_shard.dim() == 1
            # Handle TP contribution of bias
            w2_weight_shard /= module.tp_size
        else:
            if w2_weight_shard.dtype in (torch.bfloat16, torch.float16,
                                         torch.float32):
                # [N, K] -> [K, N]
                w2_weight_shard, w2_scales = self._permute_mxfp4_quantize(
                    w2_weight_shard)
            else:
                # Transpose the weights to match the expected format for the Triton gemm kernel
                # [N, K] -> [K, N]
                w2_weight_shard = w2_weight_shard.transpose(0, 1).contiguous()

        dst_w2_weight.copy_(w2_weight_shard, non_blocking=True)

        return w2_scales

    def _load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor, dst_w3_w1_weight_scale: torch.Tensor,
            transpose_scales: bool):
        if transpose_scales:
            w1_weight_scale = w1_weight_scale.transpose(
                0, 1)  # (hidden_dim / 32, intermediate_dim)
            w3_weight_scale = w3_weight_scale.transpose(
                0, 1)  # (hidden_dim / 32, intermediate_dim)

        # Swapping n_alignment and k_alignment here because we have already transposed
        w1_weight_scale = shard_and_pad_tensor(
            w1_weight_scale,
            1,
            self.k_alignment // 32,
            self.n_alignment // 2,
            module.tp_size,
            module.tp_rank,
            device=dst_w3_w1_weight_scale.device)

        w3_weight_scale = shard_and_pad_tensor(
            w3_weight_scale,
            1,
            self.k_alignment // 32,
            self.n_alignment // 2,
            module.tp_size,
            module.tp_rank,
            device=dst_w3_w1_weight_scale.device)

        # (hidden_dim / 32, intermediate_dim * 2)
        combined_scale = torch.cat([w3_weight_scale, w1_weight_scale], dim=1)

        dst_w3_w1_weight_scale.copy_(combined_scale, non_blocking=True)

    def _load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                           w2_weight_scale: torch.Tensor,
                                           dst_w2_weight_scale: torch.Tensor,
                                           transpose_scales: bool):
        if transpose_scales:
            w2_weight_scale = w2_weight_scale.transpose(
                0, 1)  # (intermediate_dim / 32, hidden_dim)

        # k_alignment is divided by 32 because every 32 values share a single scale
        # Swapping n_alignment and k_alignment here because we have already transposed
        w2_weight_scale = shard_and_pad_tensor(
            w2_weight_scale,
            0,
            self.k_alignment // 32,
            self.n_alignment,
            module.tp_size,
            module.tp_rank,
            device=dst_w2_weight_scale.device)

        dst_w2_weight_scale.copy_(w2_weight_scale, non_blocking=True)

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
        device = module.w3_w1_weight.device
        tmp_w3_w1_weight_scale = torch.empty(module.fc31_dequant.shape,
                                             dtype=torch.uint8,
                                             device=device)
        tmp_w2_weight_scale = torch.empty(module.fc2_dequant.shape,
                                          dtype=torch.uint8,
                                          device=device)
        for local_slot_id, expert_id in enumerate(
                module.initial_local_expert_ids):
            try:
                if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                    w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                    w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                    w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
                    need_to_transpose_scales = True
                elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                    # Reverse-engineered from the openai modeling class
                    combined_weight_scale = weights[
                        "gate_up_proj_weight_scale"][expert_id]
                    out_dim = combined_weight_scale.shape[-1]
                    w1_weight_scale = combined_weight_scale[..., :out_dim // 2]
                    w3_weight_scale = combined_weight_scale[..., out_dim // 2:]
                    w2_weight_scale = weights[f"down_proj_weight_scale"][
                        expert_id]
                    need_to_transpose_scales = False
                else:
                    raise NotImplementedError(
                        f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                    )
            except KeyError:
                # We will use dynamic quantization
                w1_weight_scale = self.w1_scales[expert_id]
                w3_weight_scale = self.w3_scales[expert_id]
                w2_weight_scale = self.w2_scales[expert_id]
                need_to_transpose_scales = False

            expert_idx = local_slot_id

            self._load_expert_w3_w1_weight_scale_mxfp4(
                module, w1_weight_scale, w3_weight_scale,
                tmp_w3_w1_weight_scale[expert_idx], need_to_transpose_scales)

            self._load_expert_w2_weight_scale_mxfp4(
                module, w2_weight_scale, tmp_w2_weight_scale[expert_idx],
                need_to_transpose_scales)

        self.w1_scales.clear()
        self.w3_scales.clear()
        self.w2_scales.clear()

        # Scales need to be shuffled as well
        tmp_w3_w1_weight_scale = shuffle_weight_for_activation_kernel(
            tmp_w3_w1_weight_scale)

        # Handle w3_w1_weight
        tmp_w3_w1_weight, tmp_w3_w1_weight_scale = swizzle_weight_and_scale(
            module.w3_w1_weight.data, tmp_w3_w1_weight_scale)

        module._parameters.pop('w3_w1_weight', None)
        module._parameters.pop('fc31_dequant', None)
        torch.cuda.empty_cache()

        module.w3_w1_weight = tmp_w3_w1_weight
        module.fc31_dequant = tmp_w3_w1_weight_scale

        # Handle w2_weight
        tmp_w2_weight, tmp_w2_weight_scale = swizzle_weight_and_scale(
            module.w2_weight.data, tmp_w2_weight_scale)

        module._parameters.pop('w2_weight', None)
        module._parameters.pop('fc2_dequant', None)
        torch.cuda.empty_cache()

        module.w2_weight = tmp_w2_weight
        module.fc2_dequant = tmp_w2_weight_scale

        if self.activation_dtype == torch.float8_e4m3fn:
            if max_fc31_input_scale is None or max_fc2_input_scale is None:
                module.fc31_input_dequant = None
                module.fc2_input_dequant = None
            else:
                module.fc31_input_dequant.data.copy_(max_fc31_input_scale,
                                                     non_blocking=True)
                module.fc2_input_dequant.data.copy_(max_fc2_input_scale,
                                                    non_blocking=True)

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
            rdata, gather_indx, scatter_indx = TritonEPRouter()(
                expert_logits,
                top_k,
                ep=module.ep_size,
                node_idx=module.ep_rank)
        else:
            rdata, gather_indx, scatter_indx = None, None, None

        # Step 2: Gemm1
        # Setup quantization context
        def _maybe_pad_activation(hidden_states):
            k_dim = hidden_states.shape[-1]
            padded_k_dim = get_padded_size(k_dim, self.k_alignment)
            hidden_states = torch.nn.functional.pad(hidden_states,
                                                    (0, padded_k_dim - k_dim))
            return hidden_states

        if self.activation_dtype == torch.float8_e4m3fn:
            flex_ctx_1 = FlexCtx(
                lhs_data=InFlexData(scale=hidden_states_scale), )
        else:
            flex_ctx_1 = FlexCtx()
        pc1 = PrecisionConfig(weight_scale=gemm1_scales,
                              flex_ctx=flex_ctx_1,
                              allow_tf32=False,
                              out_dtype=module.dtype)

        # Call the Triton gemm kernel, which also does permutation and activation
        alpha = module.swiglu_alpha or 1.0
        beta = module.swiglu_beta or 0.0
        hidden_states = _maybe_pad_activation(hidden_states)
        if beta == 1.0:
            act = FusedActivation(
                FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn,
                        ("alpha", "limit")), (alpha, module.swiglu_limit), 2)

            act_out = matmul_ogs(hidden_states,
                                 gemm1_weights,
                                 module.w3_w1_bias if module.bias else None,
                                 rdata,
                                 gather_indx=gather_indx,
                                 precision_config=pc1,
                                 fused_activation=act)
        else:
            act_out = matmul_ogs(hidden_states,
                                 gemm1_weights,
                                 module.w3_w1_bias if module.bias else None,
                                 rdata,
                                 gather_indx=gather_indx,
                                 precision_config=pc1)
            act_out = swiglu_torch(act_out, alpha, beta, module.swiglu_limit)

        if self.activation_dtype == torch.float8_e4m3fn:
            # Quantize the activation output manually since the Triton activation kernel doesn't support bf16 in fp8 out
            if module.fc2_input_dequant is None:
                act_out, act_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                    act_out)
            else:
                act_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    act_out, module.fc2_input_dequant)
                act_scale = module.fc2_input_dequant

        # Step 3: Gemm2
        # Setup quantization context
        if self.activation_dtype == torch.float8_e4m3fn:
            flex_ctx_2 = FlexCtx(lhs_data=InFlexData(scale=act_scale), )
        else:
            flex_ctx_2 = FlexCtx()
        pc2 = PrecisionConfig(weight_scale=gemm2_scales,
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

        def _maybe_remove_padding(gemm_output, expected_size):
            assert gemm_output.dim() == 2
            if gemm_output.shape[-1] != expected_size:
                assert gemm_output.shape[
                    -1] % self.k_alignment == 0, "The padding is not done correctly"
                gemm_output = gemm_output[:, :expected_size]
            return gemm_output

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
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
        )
        if not IS_TRITON_KERNELS_AVAILABLE:
            raise ImportError("Triton kernels are not available.")
        if torch.cuda.get_device_capability()[0] != 9 and self.ep_size > 1:
            raise NotImplementedError(
                "TritonFusedMoE is only supported on Hopper with EP size > 1.")

        assert isinstance(self.routing_method, RenormalizeMoeRoutingMethod), \
            "routing_method must be an instance of RenormalizeMoeRoutingMethod for TritonFusedMoE"
        assert not self.smart_router, "Smart router is not supported in TritonFusedMoE."

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
            assert p.shape == (self.expert_size_per_partition, ), p.shape
            assert torch.all(
                p == p[0]
            ), "All experts must have the same swiglu alpha/beta for Triton kernel"
            p = p[0].item()
            return p

        self.swiglu_alpha = _maybe_squeeze_act_param(swiglu_alpha)
        self.swiglu_beta = _maybe_squeeze_act_param(swiglu_beta)
        self.swiglu_limit = _maybe_squeeze_act_param(swiglu_limit)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_qdq():
                return TritonFP8QDQFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                return TritonMXFP4FusedMoEMethod(
                    activation_dtype=torch.float8_e4m3fn)
            elif self.quant_config.layer_quant_mode.has_w4a16_mxfp4():
                assert self.dtype in (
                    torch.bfloat16, torch.float16
                ), "Only bfloat16 and float16 are supported for w4a16_mxfp4"
                return TritonMXFP4FusedMoEMethod(activation_dtype=self.dtype)
        else:
            return TritonUnquantizedFusedMoEMethod()

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True

    def forward_impl(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert do_finalize, "TritonFusedMoE does not support do_finalize=False"
        assert use_dp_padding is None or not use_dp_padding, \
            "TritonFusedMoE does not support use_dp_padding=True"

        hidden_states = self.quant_method.apply(self, x, router_logits)

        final_hidden_states = self.reducescatter_or_allreduce(
            hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding)

        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)

    def post_load_weights(self):
        self.quant_method.post_load_weights(self)
