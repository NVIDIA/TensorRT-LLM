from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Union

import torch
from torch import nn

from tensorrt_llm import logger
from tensorrt_llm._utils import get_sm_version
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


class FusedMoEQuantScalesW4A8MXFP4FP8(NamedTuple):
    fc31_weight_block_scale: torch.Tensor
    fc31_dequant_scale: torch.Tensor
    fc2_input_scale: torch.Tensor
    fc2_weight_block_scale: torch.Tensor
    fc2_dequant_scale: torch.Tensor


def trtllmgen_maybe_get_cached_w3_w1_permute_indices(
        dst_w3_w1_weight: torch.Tensor,
        cache_permute_indices: Dict[tuple[int, int, int], torch.Tensor],
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None) -> torch.Tensor:
    if dst_w3_w1_weight.shape not in cache_permute_indices:
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
        cache_permute_indices[dst_w3_w1_weight.shape] = permute0[permute1].to(
            dst_w3_w1_weight.device)
    permute_indices = cache_permute_indices[dst_w3_w1_weight.shape]
    return permute_indices


def trtllmgen_maybe_get_cached_w2_permute_indices(
        dst_w2_weight: torch.Tensor,
        cache_permute_indices: Dict[tuple[int, int, int], torch.Tensor],
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None) -> torch.Tensor:
    if dst_w2_weight.shape not in cache_permute_indices:
        if num_elts_per_sf is None:
            permute_indices = (get_shuffle_matrix_a_row_indices(
                dst_w2_weight, epilogue_tile_m).to(dst_w2_weight.device))
        else:
            permute_indices = (get_shuffle_matrix_sf_a_row_indices(
                dst_w2_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf).to(dst_w2_weight.device))
        # Memoize permute indices as recompute is **very** costly
        cache_permute_indices[dst_w2_weight.shape] = permute_indices
    permute_indices = cache_permute_indices[dst_w2_weight.shape]
    return permute_indices


class FusedMoEMethodBase(ABC):
    """
    Base class for all fused MoE methods.
    """

    def need_load_shared_weights(self, module):
        if hasattr(
                module, "layer_load_balancer"
        ) and module.layer_load_balancer and module.layer_load_balancer.need_load_shared_weights(
        ):
            return True
        return False

    def create_weights(self,
                       module: torch.nn.Module,
                       weight_dtype: torch.dtype,
                       w3_w1_weight_shape: tuple[int, int, int],
                       w2_weight_shape: tuple[int, int, int],
                       bias_dtype: Optional[torch.dtype] = None):
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
            bias_dtype = bias_dtype or module.dtype
            w3_w1_bias = nn.Parameter(torch.empty(
                (module.expert_size_per_partition,
                 module.intermediate_size_per_partition * 2),
                dtype=bias_dtype),
                                      requires_grad=False)
            module.register_parameter("w3_w1_bias", w3_w1_bias)

            w2_bias = nn.Parameter(torch.empty(
                (module.expert_size_per_partition, module.hidden_size),
                dtype=bias_dtype),
                                   requires_grad=False)
            module.register_parameter("w2_bias", w2_bias)

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
        # Re-setup quant scales after loading weights as the tensors may have been modified.
        self.setup_quant_scales(module)

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

    def load_quant_scales(self, module: torch.nn.Module, weights: List[Dict]):
        pass

    @abstractmethod
    def setup_quant_scales(self, module: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        """
        Get the quant scales for the given slot range.
        Due to the special handling of slot_start and slot_end, we require the subclasses
        to implement this method or explicitly raise NotImplementedError.
        """
        raise NotImplementedError

    def apply(self, module: torch.nn.Module, input: torch.Tensor, *args,
              **kwargs) -> torch.Tensor:
        """
        Apply the quantization method to the input tensor.
        This isn’t necessary for all quantization methods, but it’s useful for
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
        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)
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

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        return tuple()


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

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        return FusedMoEQuantScalesFP8(
            fc1_dequant=module.fc31_dequant[slot_start:slot_end],
            fc2_quant=module.fc2_quant,
            fc2_dequant=module.fc2_dequant[slot_start:slot_end],
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

        if get_sm_version() == 100:
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

        if get_sm_version() == 100:
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

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesDeepSeekFP8BlockScales(
            fc_weight_scales=module.w3_w1_weight_scaling_factor,
            proj_weight_scales=module.w2_weight_scaling_factor,
        )

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        assert module.smart_router
        return FusedMoEQuantScalesDeepSeekFP8BlockScales(
            fc_weight_scales=module.w3_w1_weight_scaling_factor.narrow(
                0, slot_start, slot_end - slot_start),
            proj_weight_scales=module.w2_weight_scaling_factor.narrow(
                0, slot_start, slot_end - slot_start),
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

        fc31_act_scale = nn.Parameter(torch.empty(1,
                                                  module.hidden_size,
                                                  dtype=module.dtype),
                                      requires_grad=False)
        module.register_parameter("fc31_act_scale", fc31_act_scale)

        fc2_act_scale = nn.Parameter(torch.empty(
            1, module.intermediate_size_per_partition, 1, dtype=module.dtype),
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

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        assert module.smart_router
        return FusedMoEQuantScalesW4A8(
            scale_1_interleaved=module.fc31_weight_scale.narrow(
                0, slot_start, slot_end - slot_start),
            scale_2_interleaved=module.fc2_weight_scale.narrow(
                0, slot_start, slot_end - slot_start),
            pre_quant_scale_1=module.fc31_act_scale.narrow(
                0, slot_start, slot_end - slot_start),
            pre_quant_scale_2=module.fc2_act_scale.narrow(
                0, slot_start, slot_end - slot_start),
            zero_1=torch.Tensor(),
            zero_2=torch.Tensor(),
            alpha_1=module.fc31_alpha.narrow(0, slot_start,
                                             slot_end - slot_start),
            alpha_2=module.fc2_alpha.narrow(0, slot_start,
                                            slot_end - slot_start),
        )

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

        if module.sm_version == 89:
            import tensorrt_llm.quantization.functional as trtllm_f

            preprocessor = trtllm_f.preprocess_weights_for_mixed_gemm
            packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
            unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8

            w31_weight_shard = packer(
                unpacker(w31_weight_shard.cpu()).T.contiguous()).to(
                    w31_weight_shard.device)
            w31_weight_shard = preprocessor(w31_weight_shard, torch.quint4x2,
                                            torch.float8_e4m3fn,
                                            89).view(dst_w3_w1_weight.shape)
        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

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

        if module.sm_version == 89:
            import tensorrt_llm.quantization.functional as trtllm_f

            preprocessor = trtllm_f.preprocess_weights_for_mixed_gemm
            packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
            unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8

            w2_weight_shard = packer(
                unpacker(w2_weight_shard.cpu()).T.contiguous()).to(
                    w2_weight_shard.device)
            w2_weight_shard = preprocessor(w2_weight_shard, torch.quint4x2,
                                           torch.float8_e4m3fn,
                                           89).view(dst_w2_weight.shape)

        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # fc31 scales
        assert (len(module.interleave) == 2)
        all_w3_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.input_scale"])
            for expert_id in module.initial_local_expert_ids
        ]
        all_w1_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.input_scale"])
            for expert_id in module.initial_local_expert_ids
        ]
        all_w3_w1_input_scales_max = torch.max(
            torch.stack(all_w3_input_scales),
            torch.stack(all_w1_input_scales)).max()
        module.fc31_act_scale.data.copy_(
            torch.ones_like(module.fc31_act_scale) *
            (1 / all_w3_w1_input_scales_max))
        module.fc31_alpha.data.copy_((torch.ones_like(module.fc31_alpha) *
                                      all_w3_w1_input_scales_max).float())

        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.weight_scale_inv"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.weight_scale_inv"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.COLUMN)
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
            load_weight_shard(weights[f"{expert_id}.w2.input_scale"])
            for expert_id in module.initial_local_expert_ids
        ]
        all_w2_input_scales_max = torch.stack(all_w2_input_scales).to(
            module.dtype).max()
        module.fc2_act_scale.data.copy_(
            torch.ones_like(module.fc2_act_scale) *
            (1 / all_w2_input_scales_max))
        module.fc2_alpha.data.copy_((torch.ones_like(module.fc2_alpha) *
                                     all_w2_input_scales_max).float())

        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.weight_scale_inv"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.ROW)
            for expert_id in module.initial_local_expert_ids
        ]
        if module.sm_version == 89:
            w2_scales = torch.stack(all_w2_scales).to(torch.float16).view(
                module.dtype)
        else:
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

    def create_weights(self, module: torch.nn.Module, weight_dtype,
                       weight_vec_size, block_scales_dtype,
                       block_scales_vec_size):

        module.scaling_vector_size = 16
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

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        assert module.smart_router
        return FusedMoEQuantScalesNVFP4(
            fc1_act_global=module.fc31_input_scale,
            fc1_weight_block=module.w3_w1_weight_scale.narrow(
                0, slot_start, slot_end - slot_start),
            fc1_global=module.fc31_alpha.narrow(0, slot_start,
                                                slot_end - slot_start),
            fc2_act_global=module.fc2_input_scale,
            fc2_weight_block=module.w2_weight_scale.narrow(
                0, slot_start, slot_end - slot_start),
            fc2_global=module.fc2_alpha.narrow(0, slot_start,
                                               slot_end - slot_start),
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
        w1_weight_scale = load_weight_shard(w1_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_scale = load_weight_shard(w3_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
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

        dst_w3_w1_weight_scale_interleaved = torch.ops.trtllm.nvfp4_block_scale_interleave(
            dst_w3_w1_weight_scale.view(float4_sf_dtype)).view(
                self.block_scales_dtype).reshape(orig_shape)

        torch.cuda.synchronize()

        dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_interleaved)

    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        w2_weight_scale = load_weight_shard(w2_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)
        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale.shape

        dst_w2_weight_scale_interleaved = torch.ops.trtllm.nvfp4_block_scale_interleave(
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

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        """
        The TRTLLM-Gen backend of FusedMoE does not use FusedMoEQuantScalesNVFP4.
        """
        raise NotImplementedError

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        w1_weight_shard = load_weight_shard(w1_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_shard = load_weight_shard(w3_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)

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
        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)

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
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        w1_weight_scale = load_weight_shard(w1_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_scale = load_weight_shard(w3_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
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
            num_elts_per_sf=16)

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

    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        w2_weight_scale = load_weight_shard(w2_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)
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
            num_elts_per_sf=16)

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


class MXFP4WeightFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module, weight_dtype,
                       weight_vec_size, block_scales_dtype,
                       block_scales_vec_size):

        module.scaling_vector_size = 32
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size // weight_vec_size)
        w2_weight_shape = (module.expert_size_per_partition, module.hidden_size,
                           module.intermediate_size_per_partition //
                           weight_vec_size)

        # column parallel
        assert module.hidden_size % (module.scaling_vector_size *
                                     block_scales_vec_size) == 0
        w3_w1_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.intermediate_size_per_partition * 2,
                        module.hidden_size // module.scaling_vector_size //
                        block_scales_vec_size,
                        dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w3_w1_weight_scale", w3_w1_weight_scale)

        # row parallel
        assert module.intermediate_size_per_partition % (
            module.scaling_vector_size * block_scales_vec_size) == 0
        w2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.hidden_size,
                        module.intermediate_size_per_partition //
                        module.scaling_vector_size // block_scales_vec_size,
                        dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w2_weight_scale", w2_weight_scale)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

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

    @abstractmethod
    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        pass


class MXFP4WeightCutlassFusedMoEMethod(MXFP4WeightFusedMoEMethod):
    weight_dtype = FUSED_MOE_MXFP4_WEIGHT_DTYPE
    block_scales_dtype = FUSED_MOE_MXFP4_WEIGHT_BLOCK_SCALE_DTYPE

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = torch.iinfo(self.block_scales_dtype).bits // 8

        super().create_weights(module, self.weight_dtype, weight_vec_size,
                               self.block_scales_dtype, block_scales_vec_size)

    def load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        w1_weight_scale = load_weight_shard(w1_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_scale = load_weight_shard(w3_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
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

        dst_w3_w1_weight_scale.copy_(
            torch.ops.tensorrt_llm.block_scale_interleave(
                dst_w3_w1_weight_scale.view(float4_sf_dtype)).view(
                    self.block_scales_dtype).reshape(orig_shape))

    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        w2_weight_scale = load_weight_shard(w2_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)
        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale.shape

        dst_w2_weight_scale.copy_(
            torch.ops.tensorrt_llm.block_scale_interleave(
                dst_w2_weight_scale.view(float4_sf_dtype)).view(
                    self.block_scales_dtype).reshape(orig_shape))


class W4A16MXFP4CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):
    pass


class W4A8MXFP4MXFP8CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):
    pass


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

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        assert module.smart_router
        return FusedMoEQuantScalesW4A8MXFP4FP8(
            fc31_weight_block_scale=module.w3_w1_weight_scale.narrow(
                0, slot_start, slot_end - slot_start),
            fc31_dequant_scale=module.fc31_input_dequant.narrow(
                0, slot_start, slot_end - slot_start),
            fc2_input_scale=module.fc2_input_scale,
            fc2_weight_block_scale=module.w2_weight_scale.narrow(
                0, slot_start, slot_end - slot_start),
            fc2_dequant_scale=module.fc2_input_dequant.narrow(
                0, slot_start, slot_end - slot_start),
        )


class MXFP4WeightTRTLLMGenFusedMoEMethod(MXFP4WeightFusedMoEMethod):
    weight_dtype = torch.uint8
    block_scales_dtype = torch.uint8

    # Cache the permute indices during weight loading to avoid recompute
    # This assumes the same input shape always results in the same permute indices
    _cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = torch.iinfo(self.block_scales_dtype).bits // 8

        super().create_weights(module, self.weight_dtype, weight_vec_size,
                               self.block_scales_dtype, block_scales_vec_size)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        """
        The TRTLLM-Gen backend of FusedMoE does not use FusedMoEQuantScales.
        """
        raise NotImplementedError

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        w1_weight_shard = load_weight_shard(w1_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_shard = load_weight_shard(w3_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)

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
        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)

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
        w1_weight_scale = load_weight_shard(w1_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_scale = load_weight_shard(w3_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
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
            num_elts_per_sf=32)

        # Shuffle the weight according to permute indices
        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale.view(float4_sf_dtype), permute_indices)

        # Assert should only be removed during debugging
        assert w3_w1_weight_scale.is_cuda, "w3_w1_weight_scale.is_cuda should be true or suffer from slow speed"
        # Interleave the weight.
        processed_w3_w1_weight_scale = torch.ops.tensorrt_llm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))
        # Copy the result into device buffer
        dst_w3_w1_weight_scale.copy_(
            processed_w3_w1_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        w2_weight_scale = load_weight_shard(w2_weight_scale, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)
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
        processed_w2_weight_scale = torch.ops.tensorrt_llm.block_scale_interleave(
            w_shuffled)
        # Copy the result into device buffer
        dst_w2_weight_scale.copy_(
            processed_w2_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))


class W4A16MXFP4TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):
    pass


class W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):
    pass


class W4A8MXFP4FP8TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):
    # Cache the permute indices during weight loading to avoid recompute
    # This assumes the same input shape always results in the same permute indices
    _cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}

    def create_weights(self, module: torch.nn.Module):
        fc31_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)

        fc2_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                         requires_grad=False)
        module.register_parameter("fc2_input_dequant", fc2_input_dequant)

        super().create_weights(module)

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

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def get_quant_scales(self, module: torch.nn.Module, slot_start,
                         slot_end) -> tuple[torch.Tensor, ...]:
        """
        The TRTLLM-Gen backend of FusedMoE does not use FusedMoEQuantScales.
        """
        raise NotImplementedError

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

        module.fc31_input_dequant.data.copy_(tmp_fc31_input_scale.max())
        # TRTLLMGen uses dynamic MXFP8 for fc2 so we need to set scales to 1.0.
        module.fc2_input_dequant.fill_(1.0)

        # Step2: Load weight block scales.
        super().load_quant_scales(module, weights)
