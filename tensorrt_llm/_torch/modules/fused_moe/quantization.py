import math
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.functional import \
    preprocess_weights_for_mixed_gemm
from tensorrt_llm.quantization.utils.fp4_utils import (
    float4_sf_dtype, get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices, get_shuffle_matrix_sf_a_row_indices)
from tensorrt_llm.quantization.utils.fp8_utils import (
    resmooth_to_fp8_e8m0, transform_sf_into_required_layout)

from ..linear import TensorParallelMode, load_weight_shard
from .interface import MoEWeightLoadingMode

# The declarations aligns with moe_kernels.h
# pack inputs into int64, e.g. 4 x bf16 input values
FUSED_MOE_NVFP4_INPUT_DTYPE = torch.int64
# pack weights into int64, e.g. 16 x nvfp4 weight values
FUSED_MOE_NVFP4_WEIGHT_DTYPE = torch.int64
FUSED_MOE_MXFP4_WEIGHT_DTYPE = torch.int64
# pack weight block scales into int32, e.g. 4 x fp8 weight values
FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE = torch.int32
FUSED_MOE_MXFP4_WEIGHT_BLOCK_SCALE_DTYPE = torch.int32


class FusedMoEQuantScalesFP8(NamedTuple):
    fc1_dequant: torch.Tensor
    fc2_quant: torch.Tensor
    fc2_dequant: torch.Tensor
    fc1_input_dequant: torch.Tensor


class FusedMoEQuantScalesDeepSeekFP8BlockScales(NamedTuple):
    fc_weight_scales: torch.Tensor
    proj_weight_scales: torch.Tensor


class FusedMoEQuantScalesNVFP4(NamedTuple):
    fc1_act_global: torch.Tensor
    fc1_weight_block: torch.Tensor
    # fc1_global_scale = 1.0 / (fc1_weight_global_scale * fc1_act_global_scale)
    fc1_global: torch.Tensor

    fc2_act_global: torch.Tensor
    fc2_weight_block: torch.Tensor
    # fc2_global_scale = 1.0 / (fc2_weight_global_scale * fc2_act_global_scale)
    fc2_global: torch.Tensor


class FusedMoEQuantScalesW4A8(NamedTuple):
    scale_1_interleaved: torch.Tensor
    scale_2_interleaved: torch.Tensor
    pre_quant_scale_1: torch.Tensor
    pre_quant_scale_2: torch.Tensor
    zero_1: torch.Tensor
    zero_2: torch.Tensor
    alpha_1: torch.Tensor
    alpha_2: torch.Tensor


class FusedMoEQuantScalesINT8WoqPerChannel(NamedTuple):
    fc31_weight_scale: torch.Tensor
    fc2_weight_scale: torch.Tensor


class FusedMoEQuantScalesW4A16MXFP4(NamedTuple):
    scale_1_interleaved: torch.Tensor
    scale_2_interleaved: torch.Tensor


class FusedMoEQuantScalesW4A8MXFP4FP8(NamedTuple):
    fc31_weight_block_scale: torch.Tensor
    fc31_dequant_scale: torch.Tensor
    fc2_input_scale: torch.Tensor
    fc2_weight_block_scale: torch.Tensor
    fc2_dequant_scale: torch.Tensor


class FusedMoEQuantScalesW4A8MXFP4MXFP8(NamedTuple):
    fc31_weight_block_scale: torch.Tensor
    fc31_dequant_scale: torch.Tensor
    fc2_weight_block_scale: torch.Tensor
    fc2_dequant_scale: torch.Tensor


def trtllmgen_maybe_get_cached_w3_w1_permute_indices(
        dst_w3_w1_weight: torch.Tensor,
        cache_permute_indices: Dict[tuple[tuple[int, int, int], str],
                                    torch.Tensor],
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None) -> torch.Tensor:
    key = (dst_w3_w1_weight.shape, "w31", int(num_elts_per_sf or -1))
    if key not in cache_permute_indices:
        # Get permute indices and chain them together
        permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(
            dst_w3_w1_weight)
        if num_elts_per_sf is None:
            permute1 = get_shuffle_matrix_a_row_indices(
                dst_w3_w1_weight, epilogue_tile_m=epilogue_tile_m)
        else:
            permute1 = get_shuffle_matrix_sf_a_row_indices(
                dst_w3_w1_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf)
        # Memoize permute indices as recompute is **very** costly
        cache_permute_indices[key] = permute0[permute1].to(
            dst_w3_w1_weight.device)
    permute_indices = cache_permute_indices[key]
    return permute_indices


def trtllmgen_maybe_get_cached_w2_permute_indices(
        dst_w2_weight: torch.Tensor,
        cache_permute_indices: Dict[tuple[tuple[int, int, int], str],
                                    torch.Tensor],
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None) -> torch.Tensor:
    key = (dst_w2_weight.shape, "w2", int(num_elts_per_sf or -1))
    if key not in cache_permute_indices:
        if num_elts_per_sf is None:
            permute_indices = (get_shuffle_matrix_a_row_indices(
                dst_w2_weight, epilogue_tile_m).to(dst_w2_weight.device))
        else:
            permute_indices = (get_shuffle_matrix_sf_a_row_indices(
                dst_w2_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf).to(dst_w2_weight.device))
        # Memoize permute indices as recompute is **very** costly
        cache_permute_indices[key] = permute_indices
    permute_indices = cache_permute_indices[key]
    return permute_indices


def maybe_pad_for_mxfp4(weight: torch.Tensor,
                        col_alignment: int,
                        row_alignment: Optional[int] = None) -> torch.Tensor:
    col_pad_size = (col_alignment - weight.shape[-1]) % col_alignment
    if row_alignment:
        row_pad_size = (row_alignment - weight.shape[-2]) % row_alignment
        weight = F.pad(weight, (0, col_pad_size, 0, row_pad_size))
    else:
        weight = F.pad(weight, (0, col_pad_size))
    return weight


class FusedMoEMethodBase(ABC):
    """
    Base class for all fused MoE methods.
    """
    weight_alignment: int = 1

    def need_load_shared_weights(self, module):
        if hasattr(
                module, "layer_load_balancer"
        ) and module.layer_load_balancer and module.layer_load_balancer.need_load_shared_weights(
        ):
            return True
        return False

    def create_weights(
        self,
        module: torch.nn.Module,
        weight_dtype: torch.dtype,
        w3_w1_weight_shape: tuple[int, int, int],
        w2_weight_shape: tuple[int, int, int],
        bias_dtype: Optional[torch.dtype] = None,
        w3_w1_bias_shape: Optional[tuple[int, int]] = None,
        w2_bias_shape: Optional[tuple[int, int]] = None,
    ):
        # Fused gate_up_proj (column parallel)
        w3_w1_weight = nn.Parameter(torch.empty(w3_w1_weight_shape,
                                                dtype=weight_dtype),
                                    requires_grad=False)
        module.register_parameter("w3_w1_weight", w3_w1_weight)

        # down_proj (row parallel)
        w2_weight = nn.Parameter(torch.empty(w2_weight_shape,
                                             dtype=weight_dtype),
                                 requires_grad=False)
        module.register_parameter("w2_weight", w2_weight)

        # bias
        if module.bias:
            if w3_w1_bias_shape is None:
                w3_w1_bias_shape = (module.expert_size_per_partition,
                                    module.intermediate_size_per_partition * 2)
            if w2_bias_shape is None:
                w2_bias_shape = (module.expert_size_per_partition,
                                 module.hidden_size)
            bias_dtype = bias_dtype or module.dtype
            w3_w1_bias = nn.Parameter(torch.empty(w3_w1_bias_shape,
                                                  dtype=bias_dtype),
                                      requires_grad=False)
            module.register_parameter("w3_w1_bias", w3_w1_bias)

            w2_bias = nn.Parameter(torch.empty(w2_bias_shape, dtype=bias_dtype),
                                   requires_grad=False)
            module.register_parameter("w2_bias", w2_bias)
        else:
            module.w3_w1_bias = None
            module.w2_bias = None

    def load_expert_weights_to_dst(
            self, module: torch.nn.Module, weights: List[Dict],
            weight_loading_mode: MoEWeightLoadingMode,
            load_expert_ids: List[int], dst_w3_w1_weights_tensor: torch.Tensor,
            dst_w2_weights_tensor: torch.Tensor,
            dst_w3_w1_bias_tensor: Optional[torch.Tensor],
            dst_w2_bias_tensor: Optional[torch.Tensor]):
        # Multithread weight load is superseded by prefetch_files() in model_engine.py
        # Also, threading adds overhead in order to protect shuffle index cache with critical section.
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            # expert_idx is the local slot index of current rank
            expert_idx = local_slot_id

            if weight_loading_mode in [
                    MoEWeightLoadingMode.VANILLA,
                    MoEWeightLoadingMode.W4A8_CUSTOM
            ]:
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

            self.load_expert_w3_w1_weight(module, w1_weight, w3_weight,
                                          dst_w3_w1_weights_tensor[expert_idx])

            self.load_expert_w2_weight(module, w2_weight,
                                       dst_w2_weights_tensor[expert_idx])

            if module.bias:
                self.load_expert_w3_w1_weight(
                    module, w1_bias, w3_bias,
                    dst_w3_w1_bias_tensor.data[expert_idx])

                self.load_expert_w2_weight(module, w2_bias,
                                           dst_w2_bias_tensor.data[expert_idx])

    def load_weights(self, module: torch.nn.Module, weights: List[Dict],
                     weight_loading_mode: MoEWeightLoadingMode):

        self.load_expert_weights_to_dst(
            module, weights, weight_loading_mode,
            module.initial_local_expert_ids, module.w3_w1_weight.data,
            module.w2_weight.data,
            module.w3_w1_bias.data if module.bias else None,
            module.w2_bias.data if module.bias else None)

        self.load_quant_scales(module, weights)

        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            local_shared_w3_w1_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight.data.shape[1:],
                dtype=module.w3_w1_weight.data.dtype,
                device='cpu')
            local_shared_w2_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight.data.shape[1:],
                dtype=module.w2_weight.data.dtype,
                device='cpu')
            if module.bias:
                local_shared_w3_w1_bias_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w3_w1_bias.data.shape[1:],
                    dtype=module.w3_w1_bias.data.dtype,
                    device='cpu')
                local_shared_w2_bias_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w2_bias.data.shape[1:],
                    dtype=module.w2_bias.data.dtype,
                    device='cpu')
            self.load_expert_weights_to_dst(
                module, weights, weight_loading_mode,
                local_shared_load_expert_ids, local_shared_w3_w1_tensors,
                local_shared_w2_tensors,
                local_shared_w3_w1_bias_tensors if module.bias else None,
                local_shared_w2_bias_tensors if module.bias else None)
            weight_fns = {
                'w3_w1_weight': local_shared_w3_w1_tensors,
                'w2_weight': local_shared_w2_tensors
            }
            if module.bias:
                weight_fns.update({
                    'w3_w1_bias': local_shared_w3_w1_bias_tensors,
                    'w2_bias': local_shared_w2_bias_tensors
                })
            module.register_all_parameter_slot_and_to_fix_weight_fns(weight_fns)
            module.layer_load_balancer.host_tensor_sharer.finalize_layer_weights(
            )

        if hasattr(module,
                   "layer_load_balancer") and module.layer_load_balancer:
            module.layer_load_balancer.set_initial_weight_assignments(
                module.initial_global_assignments)

    def post_load_weights(self, module: torch.nn.Module):
        # Re-setup quant scales after loading weights as the tensors may have been modified.
        self.setup_quant_scales(module)

    def load_quant_scales(self, module: torch.nn.Module, weights: List[Dict]):
        pass

    @abstractmethod
    def setup_quant_scales(self, module: torch.nn.Module):
        raise NotImplementedError

    def apply(self, module: torch.nn.Module, input: torch.Tensor, *args,
              **kwargs) -> torch.Tensor:
        """
        Apply the quantization method to the input tensor.
        This isn't necessary for all quantization methods, but it's useful for
        certain backends that can encapsulate the MoE forward function.
        """
        raise NotImplementedError

    # Helper function
    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w3_w1_weight.device
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
        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    # Helper function
    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w2_weight.device
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = module.dtype
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size)
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size,
            module.intermediate_size_per_partition,
        )
        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)
        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()


def load_expert_fc31_input_scale_fp8_qdq(w1_input_scale, w3_input_scale,
                                         dst_fc31_input_scale: torch.Tensor):
    dst_fc31_input_scale.copy_(
        max(w1_input_scale[...].reshape([]), w3_input_scale[...].reshape([])))


def load_expert_fc2_input_scale_fp8_qdq(w2_input_scale,
                                        dst_fc2_input_scale: torch.Tensor):
    dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))


def load_activation_scales_fp8_qdq(module: torch.nn.Module, weights: Dict):
    tmp_fc31_input_scale = torch.empty(module.num_experts, dtype=torch.float32)
    tmp_fc2_input_scale = torch.empty(module.num_experts, dtype=torch.float32)
    for expert_id in range(module.num_experts):
        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
            w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
            w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
        elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
            w1_input_scale = weights[f"gate_up_proj_input_scale"]
            w3_input_scale = weights[f"gate_up_proj_input_scale"]
            w2_input_scale = weights[f"down_proj_input_scale"]
        else:
            raise NotImplementedError(
                f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
            )

        load_expert_fc31_input_scale_fp8_qdq(w1_input_scale, w3_input_scale,
                                             tmp_fc31_input_scale[expert_id])

        load_expert_fc2_input_scale_fp8_qdq(w2_input_scale,
                                            tmp_fc2_input_scale[expert_id])

    # max_fc31_input_scale is the maximum of all w1 input scales and w3 input scales.
    # It's used to quantize fc31 input inside the MOE op
    max_fc31_input_scale = tmp_fc31_input_scale.max()
    # max_fc2_input_scale is the maximum of all w2 input scales.
    max_fc2_input_scale = tmp_fc2_input_scale.max()

    return max_fc31_input_scale, max_fc2_input_scale


def requantize_expert_w3_w1_weight_fp8_qdq(module: torch.nn.Module,
                                           w1_weight_scale, w3_weight_scale,
                                           dst_w3_w1_weight: torch.Tensor):
    w1_weight_scale = w1_weight_scale[...].reshape([])
    w3_weight_scale = w3_weight_scale[...].reshape([])
    max_w3_w1_weight_scale = max(w1_weight_scale, w3_weight_scale)

    w3_weight = dst_w3_w1_weight.narrow(
        dim=0, start=0,
        length=module.intermediate_size_per_partition).to(dtype=module.dtype)
    w1_weight = dst_w3_w1_weight.narrow(
        dim=0,
        start=module.intermediate_size_per_partition,
        length=module.intermediate_size_per_partition).to(dtype=module.dtype)
    dequant_w3_weight = w3_weight * w3_weight_scale
    dequant_w1_weight = w1_weight * w1_weight_scale
    requant_w3_weight = (dequant_w3_weight / max_w3_w1_weight_scale).to(
        torch.float8_e4m3fn)
    requant_w1_weight = (dequant_w1_weight / max_w3_w1_weight_scale).to(
        torch.float8_e4m3fn)

    dst_w3_w1_weight.narrow(
        dim=0, start=0,
        length=module.intermediate_size_per_partition).copy_(requant_w3_weight)
    dst_w3_w1_weight.narrow(
        dim=0,
        start=module.intermediate_size_per_partition,
        length=module.intermediate_size_per_partition).copy_(requant_w1_weight)


class FP8QDQFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = torch.float8_e4m3fn

        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size)
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size,
            module.intermediate_size_per_partition,
        )
        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        fc31_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                    requires_grad=False)
        module.register_parameter("fc31_dequant", fc31_dequant)

        fc2_dequant = nn.Parameter(torch.empty(module.expert_size_per_partition,
                                               dtype=torch.float32),
                                   requires_grad=False)
        module.register_parameter("fc2_dequant", fc2_dequant)

        fc2_quant = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                 requires_grad=False)
        module.register_parameter("fc2_quant", fc2_quant)

        fc31_input_dequant = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesFP8(
            fc1_dequant=module.fc31_dequant,
            fc2_quant=module.fc2_quant,
            fc2_dequant=module.fc2_dequant,
            fc1_input_dequant=module.fc31_input_dequant,
        )

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

            requantize_expert_w3_w1_weight_fp8_qdq(
                module, w1_weight_scale, w3_weight_scale,
                module.w3_w1_weight.data[expert_idx])

            self.load_expert_w2_weight_scale_fp8(
                w2_weight_scale, tmp_w2_weight_scale[expert_idx])

        # Step3: calculate and store final loaded weights
        module.fc31_dequant.data.copy_(tmp_w3_w1_weight_scale *
                                       max_fc31_input_scale)
        module.fc2_quant.data.copy_(max_fc2_input_scale.reciprocal())
        module.fc2_dequant.data.copy_(tmp_w2_weight_scale * max_fc2_input_scale)
        module.fc31_input_dequant.data.copy_(max_fc31_input_scale)


class DeepSeekFP8BlockScalesFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = torch.float8_e4m3fn

        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size)
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size,
            module.intermediate_size_per_partition,
        )
        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        cell_div = lambda x, y: (x + y - 1) // y
        w3_w1_weight_scaling_factor = nn.Parameter(torch.empty(
            (module.expert_size_per_partition,
             cell_div(module.intermediate_size_per_partition, 128) * 2,
             cell_div(w3_w1_weight_shape[2], 128)),
            dtype=torch.float32),
                                                   requires_grad=False)
        module.register_parameter("w3_w1_weight_scaling_factor",
                                  w3_w1_weight_scaling_factor)

        w2_weight_scaling_factor = nn.Parameter(torch.empty(
            (module.expert_size_per_partition, cell_div(
                w2_weight_shape[1], 128), cell_div(w2_weight_shape[2], 128)),
            dtype=torch.float32),
                                                requires_grad=False)
        module.register_parameter("w2_weight_scaling_factor",
                                  w2_weight_scaling_factor)

        self.setup_quant_scales(module)

    def load_weights(self, module: torch.nn.Module, weights: List[Dict],
                     weight_loading_mode: MoEWeightLoadingMode):
        super().load_weights(module, weights, weight_loading_mode)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesDeepSeekFP8BlockScales(
            fc_weight_scales=module.w3_w1_weight_scaling_factor,
            proj_weight_scales=module.w2_weight_scaling_factor,
        )

    def load_expert_all_weight_scale_fp8_block_scale(
            self, module: torch.nn.Module, weights: Dict,
            load_expert_ids: List[int], dst_w3_w1_weight_scale: torch.Tensor,
            dst_w2_weight_scale: torch.Tensor, device):
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w3_scale = weights['gate_up_proj_weight_scale'][
                    expert_id].transpose(0, 1).contiguous()
                w1_scale = None
                w2_scale = weights['down_proj_weight_scale'][
                    expert_id].transpose(0, 1).contiguous()
            elif module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w3_scale = weights[f"{expert_id}.w3.weight_scale_inv"]
                w1_scale = weights[f"{expert_id}.w1.weight_scale_inv"]
                w2_scale = weights[f"{expert_id}.w2.weight_scale_inv"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            w3_w1_scale_shard = load_weight_shard(w3_scale,
                                                  module.tp_size,
                                                  module.tp_rank,
                                                  TensorParallelMode.COLUMN,
                                                  device=device)

            if w1_scale is not None:
                w1_scale_shard = load_weight_shard(w1_scale,
                                                   module.tp_size,
                                                   module.tp_rank,
                                                   TensorParallelMode.COLUMN,
                                                   device=device)
                w3_w1_scale_shard = torch.cat(
                    [w3_w1_scale_shard, w1_scale_shard], dim=-2)

            dst_w3_w1_weight_scale[local_slot_id].copy_(w3_w1_scale_shard)

            w2_scale_shard = load_weight_shard(w2_scale,
                                               module.tp_size,
                                               module.tp_rank,
                                               TensorParallelMode.ROW,
                                               device=device)
            dst_w2_weight_scale[local_slot_id].copy_(w2_scale_shard)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        self.load_expert_all_weight_scale_fp8_block_scale(
            module,
            weights,
            module.initial_local_expert_ids,
            module.w3_w1_weight_scaling_factor.data,
            module.w2_weight_scaling_factor.data,
            device=torch.device("cuda"))
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            local_shared_w3_w1_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight_scaling_factor.data.shape[1:],
                dtype=module.w3_w1_weight_scaling_factor.data.dtype,
                device='cpu')
            local_shared_w2_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight_scaling_factor.data.shape[1:],
                dtype=module.w2_weight_scaling_factor.data.dtype,
                device='cpu')
            self.load_expert_all_weight_scale_fp8_block_scale(
                module,
                weights,
                local_shared_load_expert_ids,
                local_shared_w3_w1_scale_tensors,
                local_shared_w2_scale_tensors,
                device=torch.device("cpu"))
            module.register_all_parameter_slot_and_to_fix_weight_fns({
                'w3_w1_weight_scaling_factor':
                local_shared_w3_w1_scale_tensors,
                'w2_weight_scaling_factor':
                local_shared_w2_scale_tensors,
            })


class DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm(
        DeepSeekFP8BlockScalesFusedMoEMethod):

    def load_weights(self, module: torch.nn.Module, weights: List[Dict],
                     weight_loading_mode: MoEWeightLoadingMode):
        if is_sm_100f():
            expert_ids = set(module.initial_local_expert_ids)
            if self.need_load_shared_weights(module):
                expert_ids.update(
                    module.layer_load_balancer.get_load_expert_ids())
            for name in list(weights.keys()):
                if name.endswith("weight_scale_inv"):
                    if int(name.split(".")[0]) not in expert_ids:
                        continue
                    weight_name = name.replace("weight_scale_inv", "weight")
                    logger.debug(f"Resmoothing {weight_name}")
                    weight = weights[weight_name][:]
                    scale = weights[name][:]
                    weights[weight_name], weights[name] = resmooth_to_fp8_e8m0(
                        weight, scale)
        super().load_weights(module, weights, weight_loading_mode)

    def post_load_weights(self, module: torch.nn.Module):
        super().post_load_weights(module)
        if is_sm_100f():
            transfromed_w3_w1_scale = transform_sf_into_required_layout(
                module.quant_scales[0],
                mn=module.w3_w1_weight.shape[1],
                k=module.w3_w1_weight.shape[2],
                recipe=(1, 128, 128),
                num_groups=module.w3_w1_weight.shape[0],
                is_sfa=False)
            module.w3_w1_weight_scaling_factor = nn.Parameter(
                transfromed_w3_w1_scale, requires_grad=False)
            transfromed_w2_scale = transform_sf_into_required_layout(
                module.quant_scales[1],
                mn=module.w2_weight.shape[1],
                k=module.w2_weight.shape[2],
                recipe=(1, 128, 128),
                num_groups=module.w3_w1_weight.shape[0],
                is_sfa=False)
            module.w2_weight_scaling_factor = nn.Parameter(transfromed_w2_scale,
                                                           requires_grad=False)
            self.setup_quant_scales(module)


class INT8WoqPerChannelFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        module.sm_version = get_sm_version()
        module.sm_version = 80 if module.sm_version >= 90 else module.sm_version
        module.preprocessor = preprocess_weights_for_mixed_gemm

        weight_dtype = torch.int8
        if not module.quant_config.layer_quant_mode.is_int8_weight_only():
            raise NotImplementedError(
                f"Weight Only Quantization currently only supports INT8. Got: {module.quant_config.layer_quant_mode}."
            )

        # notice the weight shape for int8 weight-only is different from the original shape,
        # since the quantized weights have their own layout
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.hidden_size,
                              module.intermediate_size_per_partition * 2)
        w2_weight_shape = (module.expert_size_per_partition,
                           module.intermediate_size_per_partition,
                           module.hidden_size)

        fc31_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.intermediate_size_per_partition * 2,
            dtype=module.dtype),
                                         requires_grad=False)
        module.register_parameter("fc31_weight_scale", fc31_weight_scale)

        fc2_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size,
            dtype=module.dtype),
                                        requires_grad=False)
        module.register_parameter("fc2_weight_scale", fc2_weight_scale)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)
        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesINT8WoqPerChannel(
            fc31_weight_scale=module.fc31_weight_scale,
            fc2_weight_scale=module.fc2_weight_scale,
        )

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        """
        w1_weight_shard = load_weight_shard(w1_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_shard = load_weight_shard(w3_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)

        weight_dtype = torch.int8

        assert module.dtype in [torch.float16, torch.bfloat16], \
            f"activation dtype should be float16 or bfloat16, got {module.dtype}"
        if not module.quant_config.layer_quant_mode.is_int8_weight_only():
            raise NotImplementedError(
                f"weight dtype should be INT8. Got: {module.quant_config.layer_quant_mode}."
            )
        # preprocess the weights for mixed gemm
        w31_weight_shard = module.preprocessor(w31_weight_shard.T.contiguous(),
                                               weight_dtype, module.dtype,
                                               module.sm_version).contiguous()
        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        """
        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)

        weight_dtype = torch.int8
        if not module.quant_config.layer_quant_mode.is_int8_weight_only():
            raise NotImplementedError(
                f"Weight Only Quantization currently only supports INT8. Got: {module.quant_config.layer_quant_mode}."
            )

        # preprocess the weights for mixed gemm
        w2_weight_shard = module.preprocessor(w2_weight_shard.T.contiguous(),
                                              weight_dtype, module.dtype,
                                              module.sm_version).contiguous()
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # fc31 scales
        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.weight_scale"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.weight_scale"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in module.initial_local_expert_ids
        ]
        w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-1)
        w3_w1_scales = w3_w1_scales.to(module.dtype)
        module.fc31_weight_scale.data.copy_(w3_w1_scales.contiguous())

        # fc2 scales
        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.weight_scale"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.ROW)
            for expert_id in module.initial_local_expert_ids
        ]
        w2_scales = torch.stack(all_w2_scales).to(module.dtype)
        module.fc2_weight_scale.data.copy_(w2_scales.contiguous())


class WInt4AFP8FusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        module.sm_version = get_sm_version()
        if module.sm_version == 89:
            module.interleave = [1, 1]
        elif module.sm_version == 90:
            module.interleave = []
            for k_shape in [
                    module.hidden_size, module.intermediate_size_per_partition
            ]:
                if k_shape % 512 == 0:
                    module.interleave.append(4)
                elif k_shape % 256 == 0:
                    module.interleave.append(2)
                elif k_shape % 128 == 0:
                    module.interleave.append(1)
                else:
                    raise NotImplementedError(
                        f"K shape is required to be multiple of 128, received {k_shape}."
                    )
        else:
            raise NotImplementedError(
                f"W4AFP8 MoE is unsupported on SM{module.sm_version}.")
        weight_dtype = torch.int8
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size // 2)
        w2_weight_shape = (module.expert_size_per_partition, module.hidden_size,
                           module.intermediate_size_per_partition // 2)

        # Multiply act with reciprocal of per-channel pre_quant_scale * per-tensor input_scale
        fc31_act_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size,
            dtype=module.dtype),
                                      requires_grad=False)
        module.register_parameter("fc31_act_scale", fc31_act_scale)

        fc2_act_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.intermediate_size_per_partition,
            1,
            dtype=module.dtype),
                                     requires_grad=False)
        module.register_parameter("fc2_act_scale", fc2_act_scale)

        # col parallel
        fc31_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size // (128 * module.interleave[0]),
            module.intermediate_size_per_partition * 2 * module.interleave[0],
            dtype=module.dtype),
                                         requires_grad=False)
        module.register_parameter("fc31_weight_scale", fc31_weight_scale)

        # row parallel
        fc2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.intermediate_size_per_partition //
                        (128 * module.interleave[1]),
                        module.hidden_size * module.interleave[1],
                        dtype=module.dtype),
            requires_grad=False)
        module.register_parameter("fc2_weight_scale", fc2_weight_scale)

        # Multiply W@X with per-tensor weight_scale_2 * per-tensor input_scale.
        fc31_alpha = nn.Parameter(torch.empty(module.expert_size_per_partition,
                                              1,
                                              dtype=torch.float32),
                                  requires_grad=False)
        module.register_parameter("fc31_alpha", fc31_alpha)

        fc2_alpha = nn.Parameter(torch.empty(module.expert_size_per_partition,
                                             1,
                                             dtype=torch.float32),
                                 requires_grad=False)
        module.register_parameter("fc2_alpha", fc2_alpha)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)
        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A8(
            scale_1_interleaved=module.fc31_weight_scale,
            scale_2_interleaved=module.fc2_weight_scale,
            pre_quant_scale_1=module.fc31_act_scale,
            pre_quant_scale_2=module.fc2_act_scale,
            zero_1=torch.Tensor(),
            zero_2=torch.Tensor(),
            alpha_1=module.fc31_alpha,
            alpha_2=module.fc2_alpha,
        )

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w3_w1_weight.device
        self.device = device

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

        packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        # SM89
        if module.sm_version == 89:
            preprocessor = preprocess_weights_for_mixed_gemm

            w31_weight_shard = packer(
                unpacker(w31_weight_shard.cpu()).T.contiguous()).to(
                    w31_weight_shard.device)
            w31_weight_shard = preprocessor(w31_weight_shard, torch.quint4x2,
                                            torch.float8_e4m3fn,
                                            89).view(dst_w3_w1_weight.shape)
        # SM90 ModelOpt quantized weights
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            # Original:  [(N//2)*I4x2, K] which is two int4 elts in output dim packed into one
            # Transpose: [K, (N//2)*I4x2]
            transposed = w31_weight_shard.cpu().T.contiguous()
            # Unpack:    [K, N*I8]
            unpacked = unpacker(transposed.view(torch.int8))
            # Transpose: [N, K*I8]
            transposed = unpacked.T.contiguous()
            # Pack:      [N, (K//2)*I4x2]
            w31_weight_shard = packer(transposed)
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:
            pass
        else:
            raise NotImplementedError(
                f"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}."
            )

        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w2_weight.device
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        if module.sm_version == 89:
            preprocessor = preprocess_weights_for_mixed_gemm

            w2_weight_shard = packer(
                unpacker(w2_weight_shard.cpu().contiguous()).T.contiguous()).to(
                    w2_weight_shard.device)
            w2_weight_shard = preprocessor(w2_weight_shard, torch.quint4x2,
                                           torch.float8_e4m3fn,
                                           89).view(dst_w2_weight.shape)
        # SM90 ModelOpt quantized weights
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            # Original:  [(N//2)*I4x2, K] which is two int4 elts in output dim packed into one
            # Transpose: [K, (N//2)*I4x2]
            transposed = w2_weight_shard.cpu().T.contiguous()
            # Unpack:    [K, N*I8]
            unpacked = unpacker(transposed.view(torch.int8))
            # Transpose: [N, K*I8]
            transposed = unpacked.T.contiguous()
            # Pack:      [N, (K//2)*I4x2]
            w2_weight_shard = packer(transposed)
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:
            pass
        else:
            raise NotImplementedError(
                f"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}."
            )
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        assert self.device.type == "cuda"

        # fc31 scales
        w4a8_custom = module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM
        if w4a8_custom:
            weight_scale_name = "weight_scale_inv"
        else:
            weight_scale_name = "weight_scale"

        assert (len(module.interleave) == 2)

        # Ensure that the input_scale remains aligned across all ranks for W4A8 custom.
        input_scale_expert_ids = module.initial_local_expert_ids if not w4a8_custom else range(
            module.num_experts)
        # fc31 scales
        all_w3_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.input_scale"],
                              device=self.device)
            for expert_id in input_scale_expert_ids
        ]
        all_w1_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.input_scale"],
                              device=self.device)
            for expert_id in input_scale_expert_ids
        ]
        all_w3_w1_input_scales_max = torch.max(
            torch.stack(all_w3_input_scales),
            torch.stack(all_w1_input_scales)).max()
        if w4a8_custom:
            # In custom W4A8 ckpt, per-tensor input_scale and per-channel pre_quant_scale are fused into input_scale
            module.fc31_act_scale.data.copy_(
                torch.ones_like(module.fc31_act_scale, device=self.device) *
                (1 / all_w3_w1_input_scales_max))
            module.fc31_alpha.data.copy_(
                (torch.ones_like(module.fc31_alpha, device=self.device) *
                 all_w3_w1_input_scales_max).float())
        else:
            # In vanilla ckpt (at least from ModelOpt), per-tensor input_scale and per-channel pre_quant_scale are separately stored
            all_w3_pre_quant_scales = [
                load_weight_shard(weights[f"{expert_id}.w3.pre_quant_scale"],
                                  module.tp_size,
                                  module.tp_rank,
                                  TensorParallelMode.ROW,
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w1_pre_quant_scales = [
                load_weight_shard(weights[f"{expert_id}.w1.pre_quant_scale"],
                                  module.tp_size,
                                  module.tp_rank,
                                  TensorParallelMode.ROW,
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w3_w1_pre_quant_scales_greater = torch.max(
                torch.stack([
                    torch.stack(all_w3_pre_quant_scales),
                    torch.stack(all_w1_pre_quant_scales)
                ]).to(module.dtype),
                dim=0,
            ).values.permute(1, 0)

            all_w3_w1_input_scales_greater = torch.max(
                torch.stack([
                    torch.stack(all_w3_input_scales),
                    torch.stack(all_w1_input_scales)
                ]).to(module.dtype),
                dim=0,
            ).values

            all_w3_w1_pre_quant_scales_div_input_scales = (
                all_w3_w1_pre_quant_scales_greater *
                (1 / all_w3_w1_input_scales_greater.reshape(
                    1, module.expert_size_per_partition).float()))

            module.fc31_act_scale.data.copy_(
                all_w3_w1_pre_quant_scales_div_input_scales.permute(1, 0))
            # In vanilla ckpt (at least from ModelOpt), per-tensor weight_scale_2 is separately stored
            all_w3_weight_scale_2 = [
                load_weight_shard(weights[f"{expert_id}.w3.weight_scale_2"],
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w1_weight_scale_2 = [
                load_weight_shard(weights[f"{expert_id}.w1.weight_scale_2"],
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w3_w1_weight_scale_2 = torch.stack([
                torch.stack(all_w3_weight_scale_2),
                torch.stack(all_w1_weight_scale_2)
            ]).to(module.dtype)
            all_w3_w1_weight_scale_2_greater = torch.max(
                all_w3_w1_weight_scale_2, dim=0).values

            all_w3_w1_weight_scale_2_mul_input_scales = (
                all_w3_w1_weight_scale_2_greater.reshape(
                    module.expert_size_per_partition, 1).float() *
                all_w3_w1_input_scales_greater.reshape(
                    module.expert_size_per_partition, 1).float())
            module.fc31_alpha.data.copy_(
                all_w3_w1_weight_scale_2_mul_input_scales.reshape(
                    module.expert_size_per_partition, 1).float())

        # Per-group weight_scale
        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.{weight_scale_name}"],
                              module.tp_size,
                              module.tp_rank,
                              TensorParallelMode.COLUMN,
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.{weight_scale_name}"],
                              module.tp_size,
                              module.tp_rank,
                              TensorParallelMode.COLUMN,
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-2)
        if module.sm_version == 89:
            w3_w1_scales = all_w3_w1_scales.to(torch.float16).view(module.dtype)
        else:
            w3_w1_scales = all_w3_w1_scales.to(torch.bfloat16).view(
                module.dtype)
        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            w3_w1_scales = w3_w1_scales.permute(1, 2, 0)
            w3_w1_scales /= all_w3_w1_weight_scale_2_greater.reshape(
                module.expert_size_per_partition).float()
            w3_w1_scales = w3_w1_scales.permute(2, 0, 1)

        w3_w1_s_shape = w3_w1_scales.shape
        w3_w1_scales_interleaved = w3_w1_scales.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[1],
            (w3_w1_s_shape[2] // module.interleave[0]), module.interleave[0])
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.permute(0, 2, 1, 3)
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[2] // module.interleave[0],
            w3_w1_s_shape[1] * module.interleave[0])
        module.fc31_weight_scale.data.copy_(
            w3_w1_scales_interleaved.contiguous())

        # fc2 scales
        all_w2_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.input_scale"],
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w2_input_scales_max = torch.stack(all_w2_input_scales).to(
            module.dtype).max()

        if w4a8_custom:
            # In custom W4A8 ckpt, per-tensor input_scale and per-channel pre_quant_scale are fused into input_scale
            module.fc2_act_scale.data.copy_(
                torch.ones_like(module.fc2_act_scale, device=self.device) *
                (1 / all_w2_input_scales_max))
            # In custom W4A8 ckpt, per-tensor weight_scale_2 is fused into alpha
            module.fc2_alpha.data.copy_(
                (torch.ones_like(module.fc2_alpha, device=self.device) *
                 all_w2_input_scales_max).float())
        else:
            # In vanilla ckpt (at least from ModelOpt), per-tensor input_scale and per-channel pre_quant_scale are separately stored
            all_w2_pre_quant_scales = [
                load_weight_shard(weights[f"{expert_id}.w2.pre_quant_scale"],
                                  module.tp_size,
                                  module.tp_rank,
                                  TensorParallelMode.COLUMN,
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w2_pre_quant_scales = torch.stack(all_w2_pre_quant_scales).to(
                module.dtype)
            all_w2_input_scales = torch.stack(all_w2_input_scales).to(
                module.dtype)
            all_w2_pre_quant_scales_div_input_scales = (
                all_w2_pre_quant_scales.permute(1, 0) *
                (1 / (all_w2_input_scales.reshape(
                    module.expert_size_per_partition).float()))).permute(1, 0)
            module.fc2_act_scale.data.copy_(
                all_w2_pre_quant_scales_div_input_scales.reshape(
                    module.fc2_act_scale.shape))
            # In vanilla ckpt (at least from ModelOpt), per-tensor weight_scale_2 is separately stored
            all_w2_weight_scale_2 = [
                load_weight_shard(weights[f"{expert_id}.w2.weight_scale_2"],
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w2_weight_scale_2 = torch.stack(all_w2_weight_scale_2).to(
                module.dtype)
            all_w2_weight_scale_2_mul_input_scales = (
                all_w2_weight_scale_2.reshape(module.expert_size_per_partition,
                                              1) *
                all_w2_input_scales.reshape(module.expert_size_per_partition,
                                            1))
            module.fc2_alpha.data.copy_(all_w2_weight_scale_2_mul_input_scales)

        # Per-group weight_scale
        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.{weight_scale_name}"],
                              module.tp_size,
                              module.tp_rank,
                              TensorParallelMode.ROW,
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        if module.sm_version == 89:
            w2_scales = torch.stack(all_w2_scales).to(torch.float16).view(
                module.dtype)
        else:
            w2_scales = torch.stack(all_w2_scales).to(torch.bfloat16).view(
                module.dtype)

        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            w2_scales = w2_scales.permute(1, 2, 0)
            all_w2_weight_scale_2 = all_w2_weight_scale_2.reshape(
                module.expert_size_per_partition)
            w2_scales /= (all_w2_weight_scale_2.float())
            w2_scales = w2_scales.permute(2, 0, 1)
        w2_s_shape = w2_scales.shape
        w2_scales_interleaved = w2_scales.reshape(
            w2_s_shape[0], w2_s_shape[1],
            (w2_s_shape[2] // module.interleave[1]), module.interleave[1])
        w2_scales_interleaved = w2_scales_interleaved.permute(0, 2, 1, 3)
        w2_scales_interleaved = w2_scales_interleaved.reshape(
            w2_s_shape[0], w2_s_shape[2] // module.interleave[1],
            w2_s_shape[1] * module.interleave[1])
        module.fc2_weight_scale.data.copy_(w2_scales_interleaved.contiguous())


class WFP4A16FusedMoEMethod(FusedMoEMethodBase):

    group_size = 32

    def create_weights(self, module: torch.nn.Module):
        module.sm_version = get_sm_version()
        if module.sm_version == 90:
            module.interleave = []
            for k_shape in [
                    module.hidden_size, module.intermediate_size_per_partition
            ]:
                module.interleave.append(128 // self.group_size)
        else:
            raise NotImplementedError(
                f"WFP4A16 MoE is unsupported on SM{module.sm_version}.")
        weight_dtype = torch.uint8
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size // 2)
        w2_weight_shape = (module.expert_size_per_partition, module.hidden_size,
                           module.intermediate_size_per_partition // 2)

        # col parallel
        assert module.hidden_size % (self.group_size *
                                     module.interleave[0]) == 0
        scale_dtype = torch.uint8
        fc31_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size // (self.group_size * module.interleave[0]),
            module.intermediate_size_per_partition * 2 * module.interleave[0],
            dtype=scale_dtype),
                                         requires_grad=False)
        module.register_parameter("fc31_weight_scale", fc31_weight_scale)

        # row parallel
        assert module.intermediate_size_per_partition % (
            self.group_size * module.interleave[1]) == 0
        fc2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.intermediate_size_per_partition //
                        (self.group_size * module.interleave[1]),
                        module.hidden_size * module.interleave[1],
                        dtype=scale_dtype),
            requires_grad=False)
        module.register_parameter("fc2_weight_scale", fc2_weight_scale)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)
        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A16MXFP4(
            scale_1_interleaved=module.fc31_weight_scale,
            scale_2_interleaved=module.fc2_weight_scale)

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w3_w1_weight.device
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

        pad_size_inter = module.intermediate_size_per_partition - w3_weight_shard.shape[
            0]
        if w3_weight_shard.ndim == 2:
            pad_size_hidden = module.hidden_size // 2 - w3_weight_shard.shape[1]
            pad_shape = (0, pad_size_hidden, 0, pad_size_inter)
        elif w3_weight_shard.ndim == 1:
            pad_shape = (0, pad_size_inter)
        else:
            raise NotImplementedError(
                f"Invalid shape of w1_weight_shard {w1_weight_shard.shape} and w3_weight_shard {w1_weight_shard.shape}"
            )

        w1_weight_shard = torch.nn.functional.pad(w1_weight_shard, pad_shape)
        w3_weight_shard = torch.nn.functional.pad(w3_weight_shard, pad_shape)

        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)

        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w2_weight.device
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        pad_size_hidden = module.hidden_size - w2_weight_shard.shape[0]
        if w2_weight_shard.ndim == 2:
            pad_size_inter = module.intermediate_size_per_partition // 2 - w2_weight_shard.shape[
                1]
            pad_shape = (0, pad_size_inter, 0, pad_size_hidden)
        elif w2_weight_shard.ndim == 1:
            pad_shape = (0, pad_size_hidden)
        else:
            raise NotImplementedError(
                f"Invalid shape of w2_weight_shard {w2_weight_shard.shape}")

        w2_weight_shard = torch.nn.functional.pad(w2_weight_shard, pad_shape)
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        device = module.fc31_weight_scale.data.device

        # fc31 scales
        assert (len(module.interleave) == 2)

        all_w3_scales = []
        all_w1_scales = []
        for expert_id in module.initial_local_expert_ids:
            w3_scale_shard = load_weight_shard(
                weights[f"{expert_id}.w3.weight_scale_inv"],
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.COLUMN,
                device=device)
            w1_scale_shard = load_weight_shard(
                weights[f"{expert_id}.w1.weight_scale_inv"],
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.COLUMN,
                device=device)

            pad_size_hidden = module.hidden_size // self.group_size - w3_scale_shard.shape[
                1]
            pad_size_inter = module.intermediate_size_per_partition - w3_scale_shard.shape[
                0]
            w3_scale_shard = torch.nn.functional.pad(
                w3_scale_shard, (0, pad_size_hidden, 0, pad_size_inter))
            w1_scale_shard = torch.nn.functional.pad(
                w1_scale_shard, (0, pad_size_hidden, 0, pad_size_inter))

            all_w3_scales.append(w3_scale_shard)
            all_w1_scales.append(w1_scale_shard)

        all_w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-2)

        w3_w1_scales = all_w3_w1_scales.to(torch.bfloat16).view(module.dtype)
        w3_w1_s_shape = w3_w1_scales.shape
        w3_w1_scales_interleaved = w3_w1_scales.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[1],
            (w3_w1_s_shape[2] // module.interleave[0]), module.interleave[0])
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.permute(0, 2, 1, 3)
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[2] // module.interleave[0],
            w3_w1_s_shape[1] * module.interleave[0])
        module.fc31_weight_scale.data.copy_(
            w3_w1_scales_interleaved.contiguous())

        # fc2 scales
        all_w2_scales = []
        for expert_id in module.initial_local_expert_ids:
            w2_scales_shard = load_weight_shard(
                weights[f"{expert_id}.w2.weight_scale_inv"],
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.ROW,
                device=device)
            pad_size_hidden = module.hidden_size - w2_scales_shard.shape[0]
            pad_size_inter = module.intermediate_size_per_partition // self.group_size - w2_scales_shard.shape[
                1]
            w2_scales_shard = torch.nn.functional.pad(
                w2_scales_shard, (0, pad_size_inter, 0, pad_size_hidden))
            all_w2_scales.append(w2_scales_shard)

        w2_scales = torch.stack(all_w2_scales).to(torch.bfloat16).view(
            module.dtype)
        w2_s_shape = w2_scales.shape
        w2_scales_interleaved = w2_scales.reshape(
            w2_s_shape[0], w2_s_shape[1],
            (w2_s_shape[2] // module.interleave[1]), module.interleave[1])
        w2_scales_interleaved = w2_scales_interleaved.permute(0, 2, 1, 3)
        w2_scales_interleaved = w2_scales_interleaved.reshape(
            w2_s_shape[0], w2_s_shape[2] // module.interleave[1],
            w2_s_shape[1] * module.interleave[1])
        module.fc2_weight_scale.data.copy_(w2_scales_interleaved.contiguous())


class NVFP4FusedMoEMethod(FusedMoEMethodBase):
    """
    Base class for NVFP4 fused MoE methods for all backends.
    """

    def create_weights(self,
                       module: torch.nn.Module,
                       weight_dtype,
                       weight_vec_size,
                       block_scales_dtype,
                       block_scales_vec_size,
                       scaling_vector_size=16):

        module.scaling_vector_size = scaling_vector_size
        # Divide by 16 because we use int64 to pack 16 fp4 values
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size // weight_vec_size)
        w2_weight_shape = (module.expert_size_per_partition, module.hidden_size,
                           module.intermediate_size_per_partition //
                           weight_vec_size)

        # Divide by 4 because we use int32 to pack 4 fp8 values
        # column parallel
        w3_w1_weight_scale = nn.Parameter(
            torch.ones(module.expert_size_per_partition,
                       module.intermediate_size_per_partition * 2,
                       module.hidden_size // module.scaling_vector_size //
                       block_scales_vec_size,
                       dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w3_w1_weight_scale", w3_w1_weight_scale)

        # row parallel
        w2_weight_scale = nn.Parameter(
            torch.ones(module.expert_size_per_partition,
                       module.hidden_size,
                       module.intermediate_size_per_partition //
                       module.scaling_vector_size // block_scales_vec_size,
                       dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w2_weight_scale", w2_weight_scale)

        fc31_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                        requires_grad=False)
        module.register_parameter("fc31_input_scale", fc31_input_scale)

        fc2_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.register_parameter("fc2_input_scale", fc2_input_scale)

        fc31_alpha = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                             dtype=torch.float32),
                                  requires_grad=False)
        module.register_parameter("fc31_alpha", fc31_alpha)

        fc2_alpha = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                            dtype=torch.float32),
                                 requires_grad=False)
        module.register_parameter("fc2_alpha", fc2_alpha)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        self.setup_quant_scales(module)

    @abstractmethod
    def load_expert_w3_w1_weight_scale_nvfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        pass

    @abstractmethod
    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        pass

    def load_expert_fc31_input_scale_nvfp4(self, w1_input_scale, w3_input_scale,
                                           dst_fc31_input_scale: torch.Tensor):
        w1_input_scale = w1_input_scale[...].reshape([])
        w3_input_scale = w3_input_scale[...].reshape([])
        assert torch.allclose(
            w1_input_scale, w3_input_scale), "w1_input_scale != w3_input_scale"
        dst_fc31_input_scale.copy_(w1_input_scale)

    def load_expert_fc2_input_scale_nvfp4(self, w2_input_scale,
                                          dst_fc2_input_scale: torch.Tensor):
        dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

    def load_expert_fc31_alpha_nvfp4(self, w1_weight_scale_2, w3_weight_scale_2,
                                     final_fc31_input_scale: torch.Tensor,
                                     dst_fc31_alpha: torch.Tensor):
        w1_weight_scale_2 = w1_weight_scale_2[...].reshape([])
        w3_weight_scale_2 = w3_weight_scale_2[...].reshape([])
        assert torch.allclose(
            w1_weight_scale_2,
            w3_weight_scale_2), "w1_weight_scale_2 != w3_weight_scale_2"

        w3_w1_weight_scale_2 = 1.0 / w1_weight_scale_2
        dst_fc31_alpha.copy_(1.0 /
                             (final_fc31_input_scale * w3_w1_weight_scale_2))

    def load_expert_fc2_alpha_nvfp4(self, w2_weight_scale_2,
                                    final_fc2_input_scale: torch.Tensor,
                                    dst_w2_alpha: torch.Tensor):
        w2_weight_scale_2 = 1.0 / w2_weight_scale_2[...].reshape([])
        dst_w2_alpha.copy_(1.0 / (final_fc2_input_scale * w2_weight_scale_2))

    def load_all_fp4_weight_scales_and_alphas(
            self, module: torch.nn.Module, weights: Dict,
            load_expert_ids: List[int], dst_w3_w1_weight_scale: torch.Tensor,
            dst_w2_weight_scale: torch.Tensor, dst_fc31_alpha: torch.Tensor,
            dst_fc2_alpha: torch.Tensor):
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
                w1_weight_scale_2 = weights[f"{expert_id}.w1.weight_scale_2"]
                w3_weight_scale_2 = weights[f"{expert_id}.w3.weight_scale_2"]
                w2_weight_scale_2 = weights[f"{expert_id}.w2.weight_scale_2"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_w3_weight_scale = weights["gate_up_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
                w1_weight_scale, w3_weight_scale = w1_w3_weight_scale.chunk(
                    2, dim=0)
                w2_weight_scale = weights["down_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
                w1_weight_scale_2 = weights["gate_up_proj_weight_scale_2"]
                w3_weight_scale_2 = weights["gate_up_proj_weight_scale_2"]
                w2_weight_scale_2 = weights["down_proj_weight_scale_2"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            if not torch.allclose(w1_weight_scale_2, w3_weight_scale_2):
                logger.warning(
                    f"w1_weight_scale_2 != w3_weight_scale_2 ({w1_weight_scale_2} != {w3_weight_scale_2}), selecting the larger value. Accuracy may be affected."
                )
                w1_weight_scale_2 = torch.max(w1_weight_scale_2,
                                              w3_weight_scale_2)
                w3_weight_scale_2 = w1_weight_scale_2

            self.load_expert_w3_w1_weight_scale_nvfp4(
                module, w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight_scale[expert_idx])
            self.load_expert_w2_weight_scale_nvfp4(
                module, w2_weight_scale, dst_w2_weight_scale[expert_idx])

            self.load_expert_fc31_alpha_nvfp4(w1_weight_scale_2,
                                              w3_weight_scale_2,
                                              module.fc31_input_scale.data,
                                              dst_fc31_alpha[expert_idx])
            self.load_expert_fc2_alpha_nvfp4(w2_weight_scale_2,
                                             module.fc2_input_scale.data,
                                             dst_fc2_alpha[expert_idx])

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(module.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(module.num_experts,
                                          dtype=torch.float32)

        for expert_id in range(module.num_experts):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights["gate_up_proj_input_scale"]
                w3_input_scale = weights["gate_up_proj_input_scale"]
                w2_input_scale = weights["down_proj_input_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            self.load_expert_fc31_input_scale_nvfp4(
                w1_input_scale, w3_input_scale, tmp_fc31_input_scale[expert_id])
            self.load_expert_fc2_input_scale_nvfp4(
                w2_input_scale, tmp_fc2_input_scale[expert_id])

        # fc31_input_scale is the reciprocal of the maximum of all w1 input scales and w3 input scales.
        module.fc31_input_scale.data.copy_(
            tmp_fc31_input_scale.max().reciprocal())
        # fc2_input_scale is the reciprocal of the maximum of all w2 input scales.
        module.fc2_input_scale.data.copy_(
            tmp_fc2_input_scale.max().reciprocal())

        # Step2: Load weight block scales and alphas.
        self.load_all_fp4_weight_scales_and_alphas(
            module, weights, module.initial_local_expert_ids,
            module.w3_w1_weight_scale.data, module.w2_weight_scale.data,
            module.fc31_alpha.data, module.fc2_alpha.data)

        # Step 3: if needed, load into shared
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            local_shared_w3_w1_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight_scale.data.shape[1:],
                dtype=module.w3_w1_weight_scale.data.dtype,
                device='cpu')
            local_shared_w2_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight_scale.data.shape[1:],
                dtype=module.w2_weight_scale.data.dtype,
                device='cpu')
            local_shared_fc31_alpha_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.fc31_alpha.data.shape[1:],
                dtype=module.fc31_alpha.data.dtype,
                device='cpu')
            local_shared_fc2_alpha_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.fc2_alpha.data.shape[1:],
                dtype=module.fc2_alpha.data.dtype,
                device='cpu')
            self.load_all_fp4_weight_scales_and_alphas(
                module, weights, local_shared_load_expert_ids,
                local_shared_w3_w1_scale_tensors, local_shared_w2_scale_tensors,
                local_shared_fc31_alpha_tensors, local_shared_fc2_alpha_tensors)

            module.register_all_parameter_slot_and_to_fix_weight_fns({
                'w3_w1_weight_scale':
                local_shared_w3_w1_scale_tensors,
                'w2_weight_scale':
                local_shared_w2_scale_tensors,
                'fc31_alpha':
                local_shared_fc31_alpha_tensors,
                'fc2_alpha':
                local_shared_fc2_alpha_tensors,
            })

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesNVFP4(
            fc1_act_global=module.fc31_input_scale,
            fc1_weight_block=module.w3_w1_weight_scale,
            fc1_global=module.fc31_alpha,
            fc2_act_global=module.fc2_input_scale,
            fc2_weight_block=module.w2_weight_scale,
            fc2_global=module.fc2_alpha,
        )


class NVFP4CutlassFusedMoEMethod(NVFP4FusedMoEMethod):
    weight_dtype = FUSED_MOE_NVFP4_WEIGHT_DTYPE
    block_scales_dtype = FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = torch.iinfo(self.block_scales_dtype).bits // 8

        super().create_weights(module, self.weight_dtype, weight_vec_size,
                               self.block_scales_dtype, block_scales_vec_size)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w3_w1_weight_scale.device
        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        # Keep weights in device buffer
        # w3
        dst_w3_weight_scale = dst_w3_w1_weight_scale.narrow(
            dim=0, start=0, length=module.intermediate_size_per_partition)
        dst_w3_weight_scale.copy_(
            w3_weight_scale.view(dst_w3_weight_scale.dtype))

        # w1
        dst_w1_weight_scale = dst_w3_w1_weight_scale.narrow(
            dim=0,
            start=module.intermediate_size_per_partition,
            length=module.intermediate_size_per_partition)
        dst_w1_weight_scale.copy_(
            w1_weight_scale.view(dst_w1_weight_scale.dtype))

        orig_shape = dst_w3_w1_weight_scale.shape

        dst_w3_w1_weight_scale_interleaved = torch.ops.trtllm.block_scale_interleave(
            dst_w3_w1_weight_scale.view(float4_sf_dtype)).view(
                self.block_scales_dtype).reshape(orig_shape)

        torch.cuda.synchronize()

        dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_interleaved)

    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w2_weight_scale.device
        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale.shape

        dst_w2_weight_scale_interleaved = torch.ops.trtllm.block_scale_interleave(
            dst_w2_weight_scale.view(float4_sf_dtype)).view(
                self.block_scales_dtype).reshape(orig_shape)

        torch.cuda.synchronize()

        dst_w2_weight_scale.copy_(dst_w2_weight_scale_interleaved)


class NVFP4TRTLLMGenFusedMoEMethod(NVFP4FusedMoEMethod):
    weight_dtype = float4_sf_dtype
    block_scales_dtype = torch.float8_e4m3fn

    # Cache the permute indices during weight loading to avoid recompute
    # This assumes the same input shape always results in the same permute indices
    _cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = 1

        super().create_weights(module, self.weight_dtype, weight_vec_size,
                               self.block_scales_dtype, block_scales_vec_size)

        fc31_scale_c = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                               dtype=torch.float32),
                                    requires_grad=False)
        module.register_parameter("fc31_scale_c", fc31_scale_c)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
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

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w3_weight, dst_w1_weight = dst_w3_w1_weight.split(
            module.intermediate_size_per_partition, dim=0)

        dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
        dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w31_weight_shard = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight, permute_indices.to(dst_w3_w1_weight.device))

        # Copy the result into device buffer
        dst_w3_w1_weight.copy_(processed_w31_weight_shard.view(
            dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        device = dst_w2_weight.device
        assert device.type == "cuda"
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)
        # Get permuted indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w2_weight = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight, permute_indices.to(dst_w2_weight.device))

        # Copy the result into device buffer
        dst_w2_weight.copy_(processed_w2_weight.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self,
            module: torch.nn.Module,
            w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor,
            num_elts_per_sf: int = 16):
        device = dst_w3_w1_weight_scale.device
        assert device.type == "cuda"
        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        # Keep weights in device buffer
        # w3
        dst_w3_weight_scale = dst_w3_w1_weight_scale.narrow(
            dim=0, start=0, length=module.intermediate_size_per_partition)
        dst_w3_weight_scale.copy_(
            w3_weight_scale.view(dst_w3_weight_scale.dtype))

        # w1
        dst_w1_weight_scale = dst_w3_w1_weight_scale.narrow(
            dim=0,
            start=module.intermediate_size_per_partition,
            length=module.intermediate_size_per_partition)
        dst_w1_weight_scale.copy_(
            w1_weight_scale.view(dst_w1_weight_scale.dtype))

        orig_shape = dst_w3_w1_weight_scale.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_scale.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf)

        # Shuffle the weight according to permute indices
        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale.view(float4_sf_dtype), permute_indices)

        # Assert should only be removed during debugging
        assert w3_w1_weight_scale.is_cuda, "w3_w1_weight_scale.is_cuda should be true or suffer from slow speed"
        # Interleave the weight.
        processed_w3_w1_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))
        # Copy the result into device buffer
        dst_w3_w1_weight_scale.copy_(
            processed_w3_w1_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

    def load_expert_w2_weight_scale_nvfp4(self,
                                          module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor,
                                          num_elts_per_sf: int = 16):
        device = dst_w2_weight_scale.device
        assert device.type == "cuda"
        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME: read from kernel

        # Assert should only be removed during debugging
        assert dst_w2_weight_scale.is_cuda, "dst_w2_weight_scale.is_cuda should be true or suffer from slow speed"

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_scale.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf)

        # Shuffle the weight according to permute indices
        w_shuffled = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_scale.view(dtype=float4_sf_dtype), permute_indices)
        # Interleave the weight.
        processed_w2_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w_shuffled)
        # Copy the result into device buffer
        dst_w2_weight_scale.copy_(
            processed_w2_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        super().load_quant_scales(module, weights)

        # last step: load fc31_scale_c
        module.fc31_scale_c.data.copy_(module.fc2_input_scale.data *
                                       module.fc31_alpha.data,
                                       non_blocking=True)


class W4A8NVFP4FP8TRTLLMGenFusedMoEMethod(NVFP4TRTLLMGenFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = 1

        NVFP4FusedMoEMethod.create_weights(self, module, self.weight_dtype,
                                           weight_vec_size,
                                           self.block_scales_dtype,
                                           block_scales_vec_size, 32)

        fc31_scale_c = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                               dtype=torch.float32),
                                    requires_grad=False)
        module.register_parameter("fc31_scale_c", fc31_scale_c)

        self.setup_quant_scales(module)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        return super().load_expert_w3_w1_weight_scale_nvfp4(
            module, w1_weight_scale, w3_weight_scale, dst_w3_w1_weight_scale,
            32)

    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        return super().load_expert_w2_weight_scale_nvfp4(
            module, w2_weight_scale, dst_w2_weight_scale, 32)


def _get_weight_alignment(weight_alignment, scaling_vector_size, tp_size,
                          shard_dim_size):

    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    # The alignment should be the least common multiple of weight_alignment, scaling_vector_size,
    # and tp_size. scaling_vector_size and tp_size must be considered
    # to avoid fractional scaling factors.
    alignment = lcm(weight_alignment, scaling_vector_size)
    alignment = lcm(alignment, tp_size)

    # If after the alignment, the sharding dim per shard is not a multiple of weight_alignment,
    # we need to pad the weights to make it a multiple of weight_alignment.
    padded_weights_dim = math.ceil(shard_dim_size / alignment) * alignment
    per_shard = padded_weights_dim // tp_size
    if per_shard % weight_alignment != 0:
        alignment = weight_alignment * math.ceil(
            per_shard / weight_alignment) * tp_size

    return alignment


class MXFP4WeightFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self,
                       module: torch.nn.Module,
                       weight_dtype,
                       weight_vec_size,
                       block_scales_dtype,
                       block_scales_vec_size,
                       weight_alignment=1,
                       bias_dtype=None):

        def round_up(x, alignment):
            return (x + alignment - 1) // alignment * alignment

        module.scaling_vector_size = 32
        intermediate_size_per_partition_padded = round_up(
            module.intermediate_size_per_partition, weight_alignment)
        hidden_size_padded = round_up(module.hidden_size, weight_alignment)

        w3_w1_weight_shape = (module.expert_size_per_partition,
                              intermediate_size_per_partition_padded * 2,
                              hidden_size_padded // weight_vec_size)
        w2_weight_shape = (module.expert_size_per_partition, hidden_size_padded,
                           intermediate_size_per_partition_padded //
                           weight_vec_size)

        # column parallel
        assert hidden_size_padded % (module.scaling_vector_size *
                                     block_scales_vec_size) == 0
        w3_w1_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        intermediate_size_per_partition_padded * 2,
                        hidden_size_padded // module.scaling_vector_size //
                        block_scales_vec_size,
                        dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w3_w1_weight_scale", w3_w1_weight_scale)

        # row parallel
        assert intermediate_size_per_partition_padded % (
            module.scaling_vector_size * block_scales_vec_size) == 0
        w2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        hidden_size_padded,
                        intermediate_size_per_partition_padded //
                        module.scaling_vector_size // block_scales_vec_size,
                        dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w2_weight_scale", w2_weight_scale)

        w3_w1_bias_shape = (module.expert_size_per_partition,
                            intermediate_size_per_partition_padded * 2)
        w2_bias_shape = (module.expert_size_per_partition, hidden_size_padded)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape, bias_dtype, w3_w1_bias_shape,
                               w2_bias_shape)

        self.setup_quant_scales(module)

    @abstractmethod
    def load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        pass

    @abstractmethod
    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        pass

    def load_all_mxfp4_weight_scales(self, module: torch.nn.Module,
                                     weights: Dict, load_expert_ids: List[int],
                                     dst_w3_w1_weight_scale: torch.Tensor,
                                     dst_w2_weight_scale: torch.Tensor):
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_w3_weight_scale = weights["gate_up_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
                w1_weight_scale, w3_weight_scale = w1_w3_weight_scale.chunk(
                    2, dim=0)
                w2_weight_scale = weights["down_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            self.load_expert_w3_w1_weight_scale_mxfp4(
                module, w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight_scale[expert_idx])
            self.load_expert_w2_weight_scale_mxfp4(
                module, w2_weight_scale, dst_w2_weight_scale[expert_idx])

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load weight block scales.
        self.load_all_mxfp4_weight_scales(module, weights,
                                          module.initial_local_expert_ids,
                                          module.w3_w1_weight_scale.data,
                                          module.w2_weight_scale.data)

        # Step 2: if needed, load into shared
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            local_shared_w3_w1_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight_scale.data.shape[1:],
                dtype=module.w3_w1_weight_scale.data.dtype,
                device='cpu')
            local_shared_w2_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight_scale.data.shape[1:],
                dtype=module.w2_weight_scale.data.dtype,
                device='cpu')

            self.load_all_mxfp4_weight_scales(module, weights,
                                              local_shared_load_expert_ids,
                                              local_shared_w3_w1_scale_tensors,
                                              local_shared_w2_scale_tensors)

            module.register_all_parameter_slot_and_to_fix_weight_fns({
                'w3_w1_weight_scale':
                local_shared_w3_w1_scale_tensors,
                'w2_weight_scale':
                local_shared_w2_scale_tensors,
            })

    @abstractmethod
    def setup_quant_scales(self, module: torch.nn.Module):
        pass


class MXFP4WeightCutlassFusedMoEMethod(MXFP4WeightFusedMoEMethod):
    weight_dtype = FUSED_MOE_MXFP4_WEIGHT_DTYPE
    block_scales_dtype = FUSED_MOE_MXFP4_WEIGHT_BLOCK_SCALE_DTYPE
    # Cutlass MoE backend requires weight elements to be 128 aligned.
    weight_alignment = 128

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = torch.iinfo(self.block_scales_dtype).bits // 8

        super().create_weights(module, self.weight_dtype, weight_vec_size,
                               self.block_scales_dtype, block_scales_vec_size,
                               self.weight_alignment)

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        # We pad before the sharding. This is done to avoid fractional scaling factors
        # per shard.
        #
        # E.g. if we pad after the sharding, with intermediate_size = 2880,
        # tp_size = 4, scaling_vector_size = 32, each shard gets 720 elements and
        # 22.5 scaling factors. After padding, each shard gets 768 in
        # intermediate_size, and 24 in scaling factors's intermediate_size.
        # The 2nd rank will start loading the 23rd scaling factor,
        # while it should've loaded 22nd for the first 16 elements only.
        # We pad the weights before the sharding to avoid this issue.
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, w1_weight.shape[0])
        if len(w1_weight.shape) == 2:
            # Pad weights
            # We already satisfy alignment factor of 2 for we pack two MXFP4 into Uint8.
            assert w1_weight.dtype == torch.uint8
            w1_weight = maybe_pad_for_mxfp4(w1_weight,
                                            self.weight_alignment // 2,
                                            alignment)
            assert w3_weight.dtype == torch.uint8
            w3_weight = maybe_pad_for_mxfp4(w3_weight,
                                            self.weight_alignment // 2,
                                            alignment)
        else:
            # Pad bias.
            assert len(w1_weight.shape) == 1
            assert len(w3_weight.shape) == 1
            w1_weight = maybe_pad_for_mxfp4(w1_weight, alignment)
            w3_weight = maybe_pad_for_mxfp4(w3_weight, alignment)

        w1_weight_shard = load_weight_shard(w1_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_shard = load_weight_shard(w3_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)

        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)
        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    # Helper function
    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        shard_w2_weight_dim = 2 * w2_weight.shape[1] if len(
            w2_weight.shape) == 2 else w2_weight.shape[0]
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, shard_w2_weight_dim)

        if len(w2_weight.shape) == 2:
            assert w2_weight.dtype == torch.uint8
            w2_weight = maybe_pad_for_mxfp4(w2_weight, alignment // 2,
                                            self.weight_alignment)
        else:
            # Pad bias.
            assert len(w2_weight.shape) == 1
            w2_weight = maybe_pad_for_mxfp4(w2_weight, self.weight_alignment)

        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)

        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        device = dst_w3_w1_weight_scale.device

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w3_weight_scale.shape[0])

        w1_weight_scale = maybe_pad_for_mxfp4(
            w1_weight_scale,
            self.weight_alignment // module.scaling_vector_size, alignment)
        w3_weight_scale = maybe_pad_for_mxfp4(
            w3_weight_scale,
            self.weight_alignment // module.scaling_vector_size, alignment)

        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        # Keep weights in device buffer
        dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale.chunk(
            2, dim=0)
        dst_w3_weight_scale.copy_(
            w3_weight_scale.view(dst_w3_weight_scale.dtype))
        dst_w1_weight_scale.copy_(
            w1_weight_scale.view(dst_w1_weight_scale.dtype))

        orig_shape = dst_w3_w1_weight_scale.shape

        dst_w3_w1_weight_scale.copy_(
            torch.ops.trtllm.block_scale_interleave(
                dst_w3_w1_weight_scale.view(float4_sf_dtype)).view(
                    self.block_scales_dtype).reshape(orig_shape))

    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        device = dst_w2_weight_scale.device

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w2_weight_scale.shape[-1])

        w2_weight_scale = maybe_pad_for_mxfp4(
            w2_weight_scale, alignment // module.scaling_vector_size,
            self.weight_alignment)

        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale.shape

        dst_w2_weight_scale.copy_(
            torch.ops.trtllm.block_scale_interleave(
                dst_w2_weight_scale.view(float4_sf_dtype)).view(
                    self.block_scales_dtype).reshape(orig_shape))


class W4A16MXFP4CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):
    pass


class W4A8MXFP4MXFP8CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        fake_input_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                        requires_grad=False)
        module.register_parameter("fake_input_scale", fake_input_scale)

        super().create_weights(module)

        self.setup_quant_scales(module)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        module.fake_input_scale.fill_(1.)

        # Step2: Load weight block scales.
        super().load_quant_scales(module, weights)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A8MXFP4MXFP8(
            fc31_weight_block_scale=module.w3_w1_weight_scale,
            fc31_dequant_scale=module.fake_input_scale,
            fc2_weight_block_scale=module.w2_weight_scale,
            fc2_dequant_scale=module.fake_input_scale,
        )


class W4A8MXFP4FP8CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        fc31_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                        requires_grad=False)
        module.register_parameter("fc31_input_scale", fc31_input_scale)

        fc31_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)

        fc2_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.register_parameter("fc2_input_scale", fc2_input_scale)

        fc2_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                         requires_grad=False)
        module.register_parameter("fc2_input_dequant", fc2_input_dequant)

        super().create_weights(module)

        self.setup_quant_scales(module)

    def load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
            self, w1_input_scale, w3_input_scale,
            dst_fc31_input_scale: torch.Tensor):
        w1_input_scale = w1_input_scale[...].reshape([])
        assert torch.allclose(
            w1_input_scale, w3_input_scale), "w1_input_scale != w3_input_scale"
        dst_fc31_input_scale.copy_(w1_input_scale)

    def load_expert_fc2_input_scale_w4a8_mxfp4_fp8(
            self, w2_input_scale, dst_fc2_input_scale: torch.Tensor):
        dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(module.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(module.num_experts,
                                          dtype=torch.float32)

        for expert_id in range(module.num_experts):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights["gate_up_proj_input_scale"]
                w3_input_scale = weights["gate_up_proj_input_scale"]
                w2_input_scale = weights["down_proj_input_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            self.load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
                w1_input_scale, w3_input_scale, tmp_fc31_input_scale[expert_id])
            self.load_expert_fc2_input_scale_w4a8_mxfp4_fp8(
                w2_input_scale, tmp_fc2_input_scale[expert_id])

        # fc31_input_scale is the reciprocal of the maximum of all w1 input scales and w3 input scales.
        module.fc31_input_scale.data.copy_(
            tmp_fc31_input_scale.max().reciprocal())
        module.fc31_input_dequant.data.copy_(tmp_fc31_input_scale.max())
        # fc2_input_scale is the reciprocal of the maximum of all w2 input scales.
        module.fc2_input_scale.data.copy_(
            tmp_fc2_input_scale.max().reciprocal())
        module.fc2_input_dequant.data.copy_(tmp_fc2_input_scale.max())

        # Step2: Load weight block scales.
        super().load_quant_scales(module, weights)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A8MXFP4FP8(
            fc31_weight_block_scale=module.w3_w1_weight_scale,
            fc31_dequant_scale=module.fc31_input_dequant,
            fc2_input_scale=module.fc2_input_scale,
            fc2_weight_block_scale=module.w2_weight_scale,
            fc2_dequant_scale=module.fc2_input_dequant,
        )


class MXFP4WeightTRTLLMGenFusedMoEMethod(MXFP4WeightFusedMoEMethod):
    weight_dtype = torch.uint8
    block_scales_dtype = torch.uint8
    # TRTLLM-Gen backend requires weight elements to be 256 aligned.
    weight_alignment = 256

    # Cache the permute indices during weight loading to avoid recompute
    # This assumes the same input shape always results in the same permute indices
    _cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = torch.iinfo(self.block_scales_dtype).bits // 8

        super().create_weights(module,
                               self.weight_dtype,
                               weight_vec_size,
                               self.block_scales_dtype,
                               block_scales_vec_size,
                               self.weight_alignment,
                               bias_dtype=torch.float32)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        device = dst_w3_w1_weight.device
        assert device.type == "cuda"

        # We pad before the sharding. This is done to avoid fractional scaling factors
        # per shard.
        #
        # E.g. if we pad after the sharding, with intermediate_size = 2880,
        # tp_size = 4, scaling_vector_size = 32, each shard gets 720 elements and
        # 22.5 scaling factors. After padding, each shard gets 768 in
        # intermediate_size, and 24 in scaling factors's intermediate_size.
        # The 2nd rank will start loading the 23rd scaling factor,
        # while it should've loaded 22nd for the first 16 elements only.
        # We pad the weights before the sharding to avoid this issue.
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, w1_weight.shape[0])
        if len(w1_weight.shape) == 2:
            # Pad weights
            # We already satisfy alignment factor of 2 for we pack two MXFP4 into Uint8.
            assert w1_weight.dtype == torch.uint8
            w1_weight = maybe_pad_for_mxfp4(w1_weight,
                                            self.weight_alignment // 2,
                                            alignment)
            assert w3_weight.dtype == torch.uint8
            w3_weight = maybe_pad_for_mxfp4(w3_weight,
                                            self.weight_alignment // 2,
                                            alignment)
        else:
            # Pad bias, TRTLLM backend expects float32 bias.
            assert len(w1_weight.shape) == 1
            assert len(w3_weight.shape) == 1
            w1_weight = maybe_pad_for_mxfp4(w1_weight, alignment).float()
            w3_weight = maybe_pad_for_mxfp4(w3_weight, alignment).float()

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
        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w3_weight, dst_w1_weight = dst_w3_w1_weight.chunk(2, dim=0)
        dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
        dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w31_weight_shard = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight, permute_indices.to(dst_w3_w1_weight.device))

        # Copy the result into device buffer
        dst_w3_w1_weight.copy_(processed_w31_weight_shard.view(
            dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        device = dst_w2_weight.device
        assert device.type == "cuda"

        shard_w2_weight_dim = 2 * w2_weight.shape[1] if len(
            w2_weight.shape) == 2 else w2_weight.shape[0]
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, shard_w2_weight_dim)

        if len(w2_weight.shape) == 2:
            assert w2_weight.dtype == torch.uint8
            w2_weight = maybe_pad_for_mxfp4(w2_weight, alignment // 2,
                                            self.weight_alignment)
        else:
            # Pad bias, TRTLLM backend expects float32 bias.
            # Divide bias by tp_size as we shard along the hidden dimension.
            # The bias is applied at each TP rank before the final accumulation.
            assert len(w2_weight.shape) == 1
            w2_weight = maybe_pad_for_mxfp4(
                w2_weight, self.weight_alignment).float() / module.tp_size

        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)
        # Get permuted indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w2_weight = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight, permute_indices.to(dst_w2_weight.device))

        # Copy the result into device buffer
        dst_w2_weight.copy_(processed_w2_weight.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        device = dst_w3_w1_weight_scale.device
        assert device.type == "cuda"

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w3_weight_scale.shape[0])

        w1_weight_scale = maybe_pad_for_mxfp4(
            w1_weight_scale,
            self.weight_alignment // module.scaling_vector_size, alignment)
        w3_weight_scale = maybe_pad_for_mxfp4(
            w3_weight_scale,
            self.weight_alignment // module.scaling_vector_size, alignment)

        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        # Keep weights in device buffer
        dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale.chunk(
            2, dim=0)
        dst_w3_weight_scale.copy_(
            w3_weight_scale.view(dst_w3_weight_scale.dtype))
        dst_w1_weight_scale.copy_(
            w1_weight_scale.view(dst_w1_weight_scale.dtype))

        orig_shape = dst_w3_w1_weight_scale.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_scale.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=32)

        # Shuffle the weight according to permute indices
        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale.view(float4_sf_dtype), permute_indices)

        # Assert should only be removed during debugging
        assert w3_w1_weight_scale.is_cuda, "w3_w1_weight_scale.is_cuda should be true or suffer from slow speed"
        # Interleave the weight.
        processed_w3_w1_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))
        # Copy the result into device buffer
        dst_w3_w1_weight_scale.copy_(
            processed_w3_w1_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        device = dst_w2_weight_scale.device
        assert device.type == "cuda"

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w2_weight_scale.shape[-1])

        w2_weight_scale = maybe_pad_for_mxfp4(
            w2_weight_scale, alignment // module.scaling_vector_size,
            self.weight_alignment)

        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME: read from kernel

        # Assert should only be removed during debugging
        assert dst_w2_weight_scale.is_cuda, "dst_w2_weight_scale.is_cuda should be true or suffer from slow speed"

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_scale.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=32)

        # Shuffle the weight according to permute indices
        w_shuffled = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_scale.view(dtype=float4_sf_dtype), permute_indices)
        # Interleave the weight.
        processed_w2_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w_shuffled)
        # Copy the result into device buffer
        dst_w2_weight_scale.copy_(
            processed_w2_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))


class W4A16MXFP4TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):
    pass


class W4A8MXFP4FP8TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        fc31_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)
        fc31_input_gate_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                               requires_grad=False)
        module.register_parameter("fc31_input_gate_dequant",
                                  fc31_input_gate_dequant)

        fc2_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                         requires_grad=False)
        module.register_parameter("fc2_input_dequant", fc2_input_dequant)

        super().create_weights(module)

    def load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
            self, w1_input_scale, w3_input_scale, w2_input_scale,
            dst_fc31_input_scale: torch.Tensor,
            dst_fc2_input_scale: torch.Tensor):
        w1_input_scale = w1_input_scale[...].reshape([])
        w2_input_scale = w2_input_scale[...].reshape([])
        assert torch.allclose(
            w1_input_scale, w3_input_scale), "w1_input_scale != w3_input_scale"
        dst_fc31_input_scale.copy_(w1_input_scale)
        dst_fc2_input_scale.copy_(w2_input_scale)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(module.num_experts,
                                           dtype=torch.float32)

        tmp_fc2_input_scale = torch.empty(module.num_experts,
                                          dtype=torch.float32)

        for expert_id in range(module.num_experts):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights["gate_up_proj_input_scale"]
                w3_input_scale = weights["gate_up_proj_input_scale"]
                w2_input_scale = weights["down_proj_input_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            self.load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
                w1_input_scale, w3_input_scale, w2_input_scale,
                tmp_fc31_input_scale[expert_id], tmp_fc2_input_scale[expert_id])

        module.fc31_input_dequant.data.copy_(tmp_fc31_input_scale.max() /
                                             tmp_fc2_input_scale.max())
        module.fc31_input_gate_dequant.data.copy_(tmp_fc31_input_scale.max())
        module.fc2_input_dequant.data.copy_(tmp_fc2_input_scale.max())

        # Step2: Load weight block scales.
        super().load_quant_scales(module, weights)


class W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        super().create_weights(module)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Load weight block scales.
        super().load_quant_scales(module, weights)
